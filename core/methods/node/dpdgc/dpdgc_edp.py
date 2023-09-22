import numpy as np
import torch
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from opacus.optimizers import DPOptimizer
from core import console
from core.args.utils import ArgInfo
from core.data.loader import SparseAdjRowLoader
from core.methods.node import DPDGC
from core.privacy.mechanisms import ComposedNoisyMechanism
from core.privacy.algorithms import PMA, NoisySGD
from core.data.transforms import BoundOutDegree
from core.modules.base import Metrics, Stage


class EdgePrivDPDGC (DPDGC):
    """edge-private DPDGC method"""

    def __init__(self,
                 num_features,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[DPDGC], exclude=['encoder_batch_norm'])]
                 ):

        super().__init__(num_features=num_features,
                         num_classes=num_classes,
                         encoder_batch_norm=False,
                         batch_size=batch_size, 
                         **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=1)

        self.adj_encoder_noisy_sgd = NoisySGD(
            noise_scale=0.0,
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size,
            epochs=self.encoder_epochs,
            max_grad_norm=self.max_grad_norm,
            replacement=True,
        )

        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[
                self.adj_encoder_noisy_sgd,
                self.pma_mechanism,
            ]
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self._adj_encoder = self.adj_encoder_noisy_sgd.prepare_module(self._adj_encoder)

    def fit(self, data: Data, learn: str = 'transductive', prefix: str = '', p: float = 0.6) -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.num_edges = data.num_edges
            self.calibrate() 

        return super().fit(data, learn=learn, prefix=prefix)

    def perturb(self, x) -> torch.Tensor:
        x = self.pma_mechanism(x, sensitivity=self.norm_scale)
        return x

    def configure_adj_encoder_optimizer(self) -> DPOptimizer:
        optimizer = super().configure_adj_encoder_optimizer()
        optimizer = self.adj_encoder_noisy_sgd.prepare_optimizer(optimizer, 2.0) # *2 for replacement DP
        return optimizer
    
    def adj_data_loader(self,data: Data, stage: Stage) -> SparseAdjRowLoader:
        dataloader = super().adj_data_loader(data,stage)
        if stage == 'train':
            dataloader.wor_sampling = True
            # dataloader.poisson_sampling = True
        return dataloader
