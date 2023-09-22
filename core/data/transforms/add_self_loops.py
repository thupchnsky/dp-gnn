from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_sparse import eye, SparseTensor


class AddSelfLoops(BaseTransform):
    def __init__(self, device: str):
        self.device = device
        
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index, _ = add_self_loops(data.edge_index)
        elif hasattr(data, 'adj_t'):
            N = data.adj_t.size(0)
            temp, _ = eye(N)
            row, col = temp
            I = SparseTensor(row=row, col=col).to(self.device)
            data.adj_t = data.adj_t + I
        else:
            raise TypeError("Not supported data type for edges!")
        return data
