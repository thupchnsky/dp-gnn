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
from core.modules.node.cm import ClassificationModule
from core.modules.node.em import EncoderModule


class GAP (NodeClassification):
    """Non-private GAP method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops:            Annotated[int,   ArgInfo(help='number of hops', option='-k')] = 2,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 encoder_layers:  Annotated[int,   ArgInfo(help='number of encoder MLP layers')] = 2,
                 base_layers:     Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 head_layers:     Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 combine:         Annotated[str,   ArgInfo(help='combination type of transformed hops', choices=MultiMLP.supported_combinations)] = 'cat',
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 encoder_epochs:  Annotated[int,   ArgInfo(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)

        if encoder_layers == 0 and encoder_epochs > 0:
            console.warning('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.encoder_epochs = encoder_epochs
        activation_fn = self.supported_activations[activation]
        
        
        self._encoder = EncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self._classifier = ClassificationModule(
            num_channels=hops+1,
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
    def classifier(self) -> ClassificationModule:
        return self._classifier

    def reset_parameters(self):
        self._encoder.reset_parameters()
        super().reset_parameters()

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '') -> Metrics:
        self.learning_setting = learn
        
        if learn == 'transductive':
            
            self.data = data.to(self.device, non_blocking=True)

            # pre-train encoder
            if self.encoder_layers > 0:
                self.data = self.pretrain_encoder(self.data, prefix=prefix)

            # compute aggregations
            self.data = self.compute_aggregations(self.data)

            # train classifier
            return super().fit(self.data, learn=learn, prefix=prefix)
        
        elif learn == 'inductive':
            
            # we first record the clone of data in self.data
            self.data = data.clone().detach().to(self.device, non_blocking=True)
            
            # in inductive setting we only know the information about the nodes in training set
            # the input data contains the complete graph information so we need to first extract the training dataset
            data = data.to(self.device, non_blocking=True)
            
            # extract the training graph (train_mask + val_mask)
            node_indices = torch.LongTensor([i for i in range(data.x.size(0))]).to(self.device)
            train_node_indices = node_indices[data.train_mask | data.val_mask]
            row, col, _ = data.adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
            
            train_edge_index, _ = subgraph(train_node_indices, edge_index, relabel_nodes=False)
            data.edge_index = train_edge_index  # change the edge index
            data = ToSparseTensor()(data)  # now data.adj_t has changed
            
            # pre-train encoder
            if self.encoder_layers > 0:
                data = self.pretrain_encoder(data, prefix=prefix)

            # compute aggregations
            data = self.compute_aggregations(data)

            # train classifier
            return super().fit(data, learn=learn, prefix=prefix)
        
        else:
            
            raise Exception('Not supported learning setting.')
    
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if (data is None or data == self.data) and self.learning_setting == 'transductive':
            data = self.data
        else:
            data.x = self._encoder.predict(data)
            data = self.compute_aggregations(data)

        return super().predict(data)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def pretrain_encoder(self, data: Data, prefix: str) -> Data:
        console.info('pretraining encoder')
        self._encoder.to(self.device)
        
        self.trainer.fit(
            model=self._encoder,
            epochs=self.encoder_epochs,
            optimizer=self.configure_encoder_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )

        self.trainer.reset()
        data.x = self._encoder.predict(data)
        return data

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x = self._normalize(x)
                x_list.append(x)

            data.x = torch.stack(x_list, dim=-1)
        return data

    def configure_encoder_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self._encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)