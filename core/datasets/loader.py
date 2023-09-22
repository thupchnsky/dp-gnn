import os
from functools import partial
from typing import Annotated
import torch
from core import console
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.datasets import Amazon as Amazon_pyg
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit
from core.args.utils import ArgInfo
from core.data.transforms import FilterClassByCount
from core.data.transforms import RemoveSelfLoops
from core.data.transforms import RemoveIsolatedNodes
from core.datasets import Facebook
from core.datasets import Amazon
from core.datasets.cSBM_dataset import dataset_ContextualSBM
from core.utils import dict2table


class DatasetLoader:
    supported_datasets = {
        'amazon': partial(Amazon, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'chameleon': partial(WikipediaNetwork,name="chameleon",
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'squirrel': partial(WikipediaNetwork,name="squirrel",
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'photo': partial(Amazon_pyg,name = 'photo',
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'computers': partial(Amazon_pyg,name = 'computers',
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'facebook': partial(Facebook, name='UIllinois20', target='year', 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=1000, remove_unlabeled=True)
            ])
        ),
        'cora': partial(Planetoid, name='Cora',
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'pubmed': partial(Planetoid, name='PubMed',
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        ),
        'cSBM': partial(dataset_ContextualSBM, epsilon=3.25, n=30000, d=10, p=2000,
            transform=Compose([
                RandomNodeSplit(num_val=0.025, num_test=0.95), 
                FilterClassByCount(min_count=50, remove_unlabeled=True)
            ])
        )
    }

    def __init__(self,
                 dataset:    Annotated[str, ArgInfo(help='name of the dataset', choices=supported_datasets)] = 'facebook',
                 data_dir:   Annotated[str, ArgInfo(help='directory to store the dataset')] = 'exp_datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False, csbm_phi=0.0) -> Data:
        if self.name == 'cSBM':
            data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name), name=f'cSBM_data_phi_{csbm_phi}', theta=csbm_phi)[0]
        else:
            data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data: Data):
        nodes_degree: torch.Tensor = data.adj_t.sum(dim=1)
        baseline: float = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
        train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{data.num_edges:,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max() + 1)}',
            'mean degree': f'{nodes_degree.mean():.2f}',
            'median degree': f'{nodes_degree.median()}',
            'train/val/test (%)': f'{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}',
            'baseline acc (%)': f'{baseline:.2f}'
        }

        table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
        console.info(table)
        console.print()
