from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_sparse import SparseTensor


class NormalizeDegree(BaseTransform):
    """
    Perform D^{-beta}AD^{beta-1} degree normalization 
    """
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            # Below implementation is for similicity. In practice, you should always use SparseTensor
            row, col = data.edge_index
            adj = SparseTensor(row=row,col=col)
            deg = adj.sum(dim=1)
            deg_inv_L = deg.pow(-self.beta)
            deg_inv_R = deg.pow(-(1-self.beta))
            deg_inv_L[deg_inv_L == float('inf')] = 0
            deg_inv_R[deg_inv_R == float('inf')] = 0
            adj = deg_inv_L.view(-1, 1) * adj * deg_inv_R.view(1, -1)
            row, col, _ = adj.coo()
            data.edge_index = torch.stack([row,col],dim=0)
            
        if hasattr(data, 'adj_t'):
            adj = data.adj_t.t()
            deg = adj.sum(dim=1)
            deg_inv_L = deg.pow(-self.beta)
            deg_inv_R = deg.pow(-(1-self.beta))
            deg_inv_L[deg_inv_L == float('inf')] = 0
            deg_inv_R[deg_inv_R == float('inf')] = 0
            adj = deg_inv_L.view(-1, 1) * adj * deg_inv_R.view(1, -1)
            data.adj_t = adj.t()
            
        return data
