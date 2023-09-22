import numpy as np
import torch
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from opacus.optimizers import DPOptimizer
from core import console
from core.args.utils import ArgInfo
from core.data.loader import SparseAdjRowLoader, NodeDataLoader
from core.methods.node import DPDGC
from core.privacy.mechanisms import ComposedNoisyMechanism
from core.privacy.algorithms import PMA, NoisySGD
from core.data.transforms import BoundOutDegree
from core.modules.base import Metrics, Stage


class KNeighborPrivDPDGC (DPDGC):
    """k-neighbor-level-private DPDGC method"""

    def __init__(self,
                 num_features,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 nk:            Annotated[int,   ArgInfo(help='max number of edges to change in adjacent dataset')] = 1,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[DPDGC], exclude=['encoder_batch_norm', 'batch_norm'])]
                 ):

        super().__init__(num_features=num_features,
                         num_classes=num_classes,
                         encoder_batch_norm=False,
                         batch_norm=False,
                         batch_size=batch_size, 
                         **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None  # will be used to set delta if it is 'auto'
        self.nk = nk
        
    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=1)

        self.adj_encoder_noisy_sgd = NoisySGD(
            noise_scale=0.0,
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size,
            epochs=self.encoder_epochs,
            max_grad_norm=self.max_grad_norm,
            replacement=True
        )

        self.classifier_noisy_sgd = NoisySGD(
            noise_scale=0.0,
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            replacement=True
        )
        
        if self.nk > 0:
            composed_mechanism = ComposedNoisyMechanism(
                noise_scale=0.0,
                mechanism_list=[
                    self.adj_encoder_noisy_sgd,
                    self.pma_mechanism,
                    self.classifier_noisy_sgd,
                ]
            )
        else:
            composed_mechanism = ComposedNoisyMechanism(
                noise_scale=0.0,
                mechanism_list=[
                    self.adj_encoder_noisy_sgd,
                    self.classifier_noisy_sgd,
                ]
            )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes))) 
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')
        
        self._adj_encoder = self.adj_encoder_noisy_sgd.prepare_module(self._adj_encoder)
        self._classifier = self.classifier_noisy_sgd.prepare_module(self._classifier)

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '', p: float = 0.6) -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate() 
    
        return super().fit(data, learn=learn, prefix=prefix)

    def perturb(self, x) -> torch.Tensor:
        # Note that the node to be replaced will be take care of by the following DPSGD. 
        # Hence, the effective sensitivity is sqrt(nk)
        if self.nk > 0:
            x = self.pma_mechanism(x, sensitivity=self.norm_scale*np.sqrt(self.nk))
        return x

    def configure_adj_encoder_optimizer(self) -> DPOptimizer:
        optimizer = super().configure_adj_encoder_optimizer()
        # noise_factor = 2 for replacement dp.
        optimizer = self.adj_encoder_noisy_sgd.prepare_optimizer(optimizer, 2.0*np.sqrt(self.nk+1))
        return optimizer
    
    def configure_optimizer(self) -> DPOptimizer:
        optimizer = super().configure_optimizer()
        optimizer = self.classifier_noisy_sgd.prepare_optimizer(optimizer, 2.0) # replacement DP
        return optimizer
    
    def adj_data_loader(self,data: Data, stage: Stage) -> SparseAdjRowLoader:
        dataloader = super().adj_data_loader(data,stage)
        if stage == 'train':
            dataloader.wor_sampling = True
        return dataloader
    
    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.wor_sampling = True
        return dataloader
