import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from torch_geometric.loader.utils import to_hetero_csc
from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader.utils import filter_hetero_data
from .data_utils import *

import numpy as np
import os
from tqdm import tqdm

# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
class IPFDataset(InMemoryDataset):
    def __init__(self, root, infile, transform=None, pre_transform=None, pre_filter=None):
        self.infile = infile
        super(IPFDataset, self).__init__(root, transform,
                                         pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.infile, 'raw')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return f'{self.infile}'

    @property
    def processed_dir(self):
        return os.path.join(self.root,
                            f'hetero_{self.infile.split(".")[0]}_processed')

    @property
    def raw_paths(self):
        return os.path.join(self.root, self.raw_file_names)

    def process(self):

        gene_info = np.load(os.path.join(self.root, self.infile))
        data_list = []
        for idx in tqdm(range(gene_info.shape[0])):
            info = gene_info[idx]
            gene_idx = int(info[0])
            gene_idx = torch.tensor([gene_idx])
            # expression vectors
            exprs = torch.tensor(np.array(info[72:232], dtype=np.float32)).view(1, -1)
            # text embeddings
            embs = torch.tensor(np.array(info[232:1000], dtype=np.float32)).view(1, -1)
            # label
            y = float(info[-1])
            y = torch.tensor(y).view(1,-1)
            data = Data(gene_idx=gene_idx, exprs=exprs, embs=embs, y=y)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


