from typing import Callable
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch.nn.parameter import Parameter
from core.models import MLP
from torch_geometric.data import Data
from torch_geometric.nn import Linear
from core.modules.node.mlp import MLPNodeClassifier, MLPClassifier
from core.modules.node.graph_mlp import GraphNodeClassifier
from torch_sparse import SparseTensor, matmul


class EncoderModule(MLPNodeClassifier):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,  
                 encoder_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
        )

        self.encoder_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=encoder_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )

        self.dropout_fn = Dropout(p=dropout, inplace=True)
        self.activation_fn = activation_fn
        self.normalize = normalize
        self.bn = BatchNorm1d(hidden_dim) if batch_norm else False

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        x = self.bn(x) if self.bn else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        x = super().forward(x)
        return x

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        x = data.x
        x = self.encoder_mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x
        
    def reset_parameters(self):
        super().reset_parameters()
        self.encoder_mlp.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()


class GraphEncoderModule(GraphNodeClassifier):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,
                 encoder_layers: int = 2, 
                 head_layers: int = 1,
                 hops: int = 5,
                 dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 mlp_inst_norm: bool = False,
                 ):

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
        )
        
        self.encoder_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=encoder_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )
        
        self.hops = hops
        self.mlp_inst_norm = mlp_inst_norm
        self.dropout_fn = Dropout(p=dropout, inplace=True)
        self.activation_fn = activation_fn
        self.bn = BatchNorm1d(hidden_dim * (hops + 1)) if batch_norm else False  # [noisy_x_emb, K propagated features]
    
    def forward(self, x: Tensor, adj_t: SparseTensor, noisy_x_emb: Tensor) -> Tensor:
        x = self.encoder_mlp(x)
        if self.mlp_inst_norm:
            x = F.normalize(x, p=2, dim=-1)
        
        Z = [noisy_x_emb]
        for _ in range(self.hops):
            x = matmul(adj_t, x)
            x = F.normalize(x, p=2, dim=-1)
            Z.append(x)
        
        x = torch.cat(Z, dim=-1) 
        x = self.bn(x) if self.bn else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        x = super().forward(x, adj_t, noisy_x_emb)
        return x

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        x = self.encoder_mlp(data.x)  # we do not perform normalization here
        return x
        
    def reset_parameters(self):
        super().reset_parameters()
        self.encoder_mlp.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()


class AdjEncoderModule(MLPClassifier):
    '''
        This encoder module is used for MLP_A, and it is similar as EncoderModule, except that now 
        we need to normalize the rows of the weight matrix W.
    '''
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,
                 head_layers: int = 1,
                 dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 batch_norm: bool = False,
                 norm_scale: float = 1.0,
                 ):

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
        )
        
        # The MLP_A is hard-coded to be a one layer MLP (XW^T+b)
        self.mlp_adj = Linear(-1, hidden_dim)
        self.norm_scale = norm_scale
        
        # Freeze the head MLP as initialization, and disable bias
        for lin in self.model.layers:
            lin.bias = torch.nn.init.constant_(lin.bias, 0.0)
            # we use random orthonormal matrix to construct projection. It prevents potential rank collapse.
            Rot = torch.randn(hidden_dim,hidden_dim)
            svd = torch.linalg.svd(Rot)
            orth = svd[0] @ svd[2]
            lin.weight = Parameter(torch.eye(num_classes,hidden_dim) @ orth * math.sqrt(3.0/hidden_dim*2))
        for param in self.model.parameters():
            param.requires_grad = False
        
        # extract the weight matrix and do column normalization of W (row normalization of W^T)
        # self.mlp_adj.state_dict()['weight'] = F.normalize(self.mlp_adj.state_dict()['weight'], p=2, dim=0)
        # debug use only
        # print(torch.norm(self.mlp_adj.state_dict()['weight'], dim=0))
        
        self.dropout_fn = Dropout(p=dropout, inplace=True)
        self.activation_fn = activation_fn
        self.bn = BatchNorm1d(hidden_dim) if batch_norm else False

    def forward(self, x: Tensor) -> Tensor:
        # Here the x would actually be data.adj_t
        x = self.mlp_adj(x)
        self.mlp_adj.state_dict()['weight'] = F.normalize(self.mlp_adj.state_dict()['weight'], p=2, dim=0) * self.norm_scale
        x = self.bn(x) if self.bn else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        x = super().forward(x)
        return x
    
    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        # Ensure weight matrix is normalized.
        self.mlp_adj.state_dict()['weight'] = F.normalize(self.mlp_adj.state_dict()['weight'], p=2, dim=0) * self.norm_scale
        x = data.adj_t.t().to_torch_sparse_coo_tensor()
        x = self.mlp_adj(x)
        return x
        
    def reset_parameters(self):
        super().reset_parameters()
        self.mlp_adj.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()
