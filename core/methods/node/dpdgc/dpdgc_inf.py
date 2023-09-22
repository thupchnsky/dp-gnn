import torch
from typing import Annotated, Optional
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node.base import NodeClassification
from core.models.multi_mlp import MultiMLP
from core.modules.base import Metrics
from core.modules.node.cm import DPDGCClassificationModule
from core.modules.node.em import AdjEncoderModule
from core.data.transforms import BoundOutDegree, BoundDegree, AddSelfLoops, NormalizeDegree


class DPDGC (NodeClassification):
    """Non-private DGC method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_dim:           Annotated[int,   ArgInfo(help='hidden dimension of the MLP layers')] = 64,
                 encoder_head_layers:  Annotated[int,   ArgInfo(help='number of MLP layers in MLP encoder')] = 1,
                 base_layers:          Annotated[int,   ArgInfo(help='number of base MLP layers in classifier')] = 2,
                 head_layers:          Annotated[int,   ArgInfo(help='number of head MLP layers in classifier')] = 1,
                 combine:              Annotated[str,   ArgInfo(help='combination type of transformed hops', choices=MultiMLP.supported_combinations)] = 'cat',
                 activation:           Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:              Annotated[float, ArgInfo(help='general dropout rate')] = 0.5,
                 batch_norm:           Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 encoder_batch_norm:   Annotated[bool,  ArgInfo(help='if true, then encoder module uses batch normalization')] = True,
                 encoder_epochs:       Annotated[int,   ArgInfo(help='number of epochs for MLP encoder pre-training (ignored if encoder_layers=0)')] = 50,
                 encoder_lr:           Annotated[float,   ArgInfo(help='learning rate for GRAPH encoder module')] = 5e-3,
                 encoder_wd:           Annotated[float,   ArgInfo(help='weight decay for GRAPH encoder module')] = 1e-5,
                 encoder_dropout:      Annotated[float,   ArgInfo(help='dropout rate for GRAPH encoder module')] = 0.0,
                 norm_scale:           Annotated[float,   ArgInfo(help='scale of MLP_A row norm')] = 1e-8,
                 **kwargs:             Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)

        # adj encoder specific learning parameters
        self.encoder_head_layers = encoder_head_layers
        self.encoder_epochs = encoder_epochs
        self.encoder_lr = encoder_lr
        self.encoder_wd = encoder_wd
        self.encoder_dropout = encoder_dropout
        self.norm_scale = norm_scale
        
        activation_fn = self.supported_activations[activation]
        
        self._adj_encoder = AdjEncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            head_layers=encoder_head_layers,
            dropout=encoder_dropout,
            activation_fn=activation_fn,
            batch_norm=encoder_batch_norm,
            norm_scale=norm_scale,
        )
        
        self._classifier = DPDGCClassificationModule(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def classifier(self) -> DPDGCClassificationModule:
        return self._classifier

    def reset_parameters(self):
        self._adj_encoder.reset_parameters()
        super().reset_parameters()

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '') -> Metrics:
        self.learning_setting = learn
        assert learn == 'transductive', 'DGC only support transductive setting for now!'
        
        # in transductive setting we will need to cache the computed data.x for test and predict purpose
        self.data = data.to(self.device, non_blocking=True)
        xA, adj_pretrain_metrics = self.pretrain_adj_encoder(self.data, prefix=prefix)
        xA = self.perturb(xA)
        console.info('adj encoder pretrain statistics')
        console.info(adj_pretrain_metrics)
        
        # concatenate the features for later use
        self.data.x = torch.cat([self.data.x, xA], dim=-1)

        return super().fit(self.data, learn=learn, prefix=prefix)  # train classifier
        
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        assert self.learning_setting == 'transductive', 'DGC only support transductive setting for now!'
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            # only in transductive we can use cached results
            data = self.data
        return super().test(data, prefix=prefix)
    
    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        assert self.learning_setting == 'transductive', 'DGC only support transductive setting for now!'
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            data = self.data
        return super().predict(data)
    
    def perturb(self, x) -> torch.Tensor:
        # do not add pma noise for non-dp methods
        return x

    def pretrain_adj_encoder(self, data: Data, prefix: str) -> Data:
        console.info('pretraining adj encoder (MLP_A)')
        self._adj_encoder.to(self.device)
        
        best_metrics = self.trainer.fit(
            model=self._adj_encoder,
            epochs=self.encoder_epochs,
            optimizer=self.configure_adj_encoder_optimizer(),
            train_dataloader=self.adj_data_loader(data, 'train'), 
            val_dataloader=self.adj_data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )
        
        self.trainer.reset()
        xA = self._adj_encoder.predict(data)
        return xA, best_metrics
    
    def configure_adj_encoder_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self._adj_encoder.parameters(), lr=self.encoder_lr, weight_decay=self.encoder_wd)
