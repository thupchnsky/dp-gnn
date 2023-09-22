from typing import Literal, Optional, Union
import torch
import numpy as np
from collections.abc import Iterator
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.transforms import ToSparseTensor
from core.modules.base import Stage
from torch_sparse import SparseTensor


class SparseDataset(Dataset):

    def __init__(self, 
                 data: Data,
                 stage: Stage):
        
        self.device = data.x.device
        self.adj_mat = data.adj_t.t().to_dense()[data[f'{stage}_mask']]
        self.label = data.y[data[f'{stage}_mask']]

    def __len__(self):
        return self.adj_mat.shape[0]

    def __getitem__(self, idx):
        return self.adj_mat[idx], self.label[idx]


def AdjDataLoader(data: Data, 
                  stage: Stage,
                  batch_size: Union[int, Literal['full']] = 'full', 
                  shuffle: bool = True, 
                  drop_last: bool = False, 
                  sampling: str = 'None'):
    
    assert sampling == 'None', 'Other sampling method is not supported for this loader!'
    
    if batch_size == 'full':
        node_indices = data[f'{stage}_mask'].nonzero().view(-1)
        batch_size = node_indices.size(0)
    
    dataset = SparseDataset(data, stage)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class SparseAdjRowLoader:
    """
    Args:
        data (Data): The graph data object.
        stage (Stage): Training stage. One of 'train', 'val', 'test'.
        batch_size (int or 'full', optional): The batch size.
            If set to 'full', the entire graph is used as a single batch.
            (default: 'full')
        hops (int, optional): The number of hops to sample neighbors.
            If set to None, all neighbors are included. (default: None)
        shuffle (bool, optional): If set to True, the nodes are shuffled
            before batching. (default: True)
        drop_last (bool, optional): If set to True, the last batch is
            dropped if it is smaller than the batch size. (default: False)
        poisson_sampling (bool, optional): If set to True, poisson sampling
            is used to sample nodes. (default: False)
        wor_sampling (bool, optional): If set to True, a random yet fixed size
            sampling is used to sample nodes. (default: False)
    """
    def __init__(self, 
                 data: Data, 
                 stage: Stage,
                 batch_size: Union[int, Literal['full']] = 'full',
                 shuffle: bool = True, 
                 drop_last: bool = False, 
                 poisson_sampling: bool = False,
                 wor_sampling: bool = False):

        self.adj = data.adj_t.t()
        self.label = data.y
        self.stage = stage
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.poisson_sampling = poisson_sampling
        self.wor_sampling = wor_sampling
        self.device = data.x.device
        self.node_indices = data[f'{stage}_mask'].nonzero().view(-1)
        self.num_nodes = self.node_indices.size(0)
        
        assert not (self.poisson_sampling and self.wor_sampling)
        
    def __iter__(self) -> Iterator[Data]:
        if self.batch_size == 'full':
            row, col, _ = self.adj[self.node_indices, :].coo()
            y = self.label[self.node_indices]
            
            # processing the row
            row[1:] = torch.diff(row)
            row[0] = 0
            row = row.to(torch.bool)
            row = torch.cumsum(row,0)
            
            yield SparseTensor(row=row, col=col, sparse_sizes=(self.num_nodes, self.label.size(0))).to_torch_sparse_coo_tensor().to(self.device), y.to(self.device)
            return

        if self.shuffle and not (self.poisson_sampling or self.wor_sampling):
            perm = torch.randperm(self.num_nodes, device=self.device)
            self.node_indices = self.node_indices[perm]

        for i in range(0, self.num_nodes, self.batch_size):
            if self.drop_last and i + self.batch_size > self.num_nodes:
                break

            if self.poisson_sampling:
                sampling_prob = self.batch_size / self.num_nodes
                sample_mask = torch.rand(self.num_nodes, device=self.device) < sampling_prob
                batch_nodes = self.node_indices[sample_mask]
            elif self.wor_sampling:
                batch_nodes = self.node_indices[torch.randperm(self.num_nodes)[:self.batch_size]]
            else:    
                batch_nodes = self.node_indices[i: min(i + self.batch_size, self.num_nodes)]
            
            
            row, col, _ = self.adj[batch_nodes, :].coo()
            y = self.label[batch_nodes]
            
            # processing the row
            row[1:] = torch.diff(row)
            row[0] = 0
            row = row.to(torch.bool)
            row = torch.cumsum(row,0)
            
            yield SparseTensor(row=row, col=col, sparse_sizes=(batch_nodes.size(0), self.label.size(0))).to_dense().to(self.device), y.to(self.device)

    def __len__(self) -> int:
        if self.batch_size == 'full':
            return 1
        elif self.drop_last:
            return self.num_nodes // self.batch_size
        else:
            return (self.num_nodes + self.batch_size - 1) // self.batch_size
