from typing import Callable, Iterable, Literal, get_args
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, LazyBatchNorm1d, BatchNorm1d, Linear, Module, Dropout
from core.models import MLP


class LINKX(Module):
    CombType = Literal['cat', 'sum', 'max', 'mean', 'att']
    supported_combinations = get_args(CombType)

    def __init__(self, *,
                 num_features: int,
                 output_dim: int,
                 hidden_dim: int = 16,
                 base_layers: int = 2,
                 head_layers: int = 1,
                 combination: CombType = 'cat',
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 plain_last: bool = True,
                 ):

        super().__init__()
        
        self.num_features = num_features
        self.combination = combination
        self.activation_fn = activation_fn
        self.dropout_fn = Dropout(dropout, inplace=True)

        self.mlp_feat = MLP(hidden_dim=hidden_dim,
                            output_dim=hidden_dim,
                            num_layers=base_layers,
                            dropout=dropout,
                            activation_fn=activation_fn,
                            batch_norm=batch_norm,
                            plain_last=True)
        
        self.mlp_W = MLP(hidden_dim=hidden_dim,
                         output_dim=hidden_dim,
                         num_layers=base_layers,
                         dropout=dropout,
                         activation_fn=activation_fn,
                         batch_norm=batch_norm,
                         plain_last=True)
        
        if combination == 'att':
            self.hidden_dim = hidden_dim
            self.num_heads = 2
            self.Q = Linear(in_features=hidden_dim, out_features=self.num_heads, bias=False)

        self.bn1 = BatchNorm1d(2*hidden_dim) if batch_norm else False
        self.bn2 = BatchNorm1d(hidden_dim) if batch_norm else False

        self.head_mlp = MLP(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=plain_last,
        )

    def forward(self, x: Tensor) -> Tensor:  
        # x_stack input is of form [raw features, embedded adj]
        xX = self.mlp_feat(x[:, 0: self.num_features])
        xA = x[:, self.num_features:]
        
        h_list = [xX, xA]
        h = self.combine(h_list)
        
        h = self.bn1(h) if self.bn1 else h
        h = self.dropout_fn(h)
        h = self.activation_fn(h)
        
        h = self.mlp_W(h)
        h = h + xX + xA
        
        h = self.bn2(h) if self.bn2 else h
        h = self.dropout_fn(h)
        h = self.activation_fn(h)
        
        h = self.head_mlp(h)
        return h

    def combine(self, h_list: Iterable[Tensor]) -> Tensor:
        if self.combination == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination == 'mean':
            return torch.stack(h_list, dim=0).mean(dim=0)
        elif self.combination == 'max':
            return torch.stack(h_list, dim=0).max(dim=0).values
        elif self.combination == 'att':
            H = torch.stack(h_list, dim=1)  # (node, hop, dim)
            W = F.leaky_relu(self.Q(H), 0.2).softmax(dim=0)  # (node, hop, head)
            out = H.transpose(1, 2).matmul(W).view(-1, self.hidden_dim * self.num_heads)
            return out
        else:
            raise ValueError(f'Unknown combination type {self.combination}')

    def reset_parameters(self):
        self.mlp_feat.reset_parameters()
        self.mlp_W.reset_parameters()
        if self.combination == 'att':
            self.Q.reset_parameters()
        if self.bn1:
            self.bn1.reset_parameters()
        if self.bn2:
            self.bn2.reset_parameters()
        self.head_mlp.reset_parameters()
