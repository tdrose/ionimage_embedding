import os.path as osp
from typing import Dict, Optional
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset

from .constants import COLOC_NET_DISCRETE_DATA
from .utils import create_edgelist


def get_GraphData(
        node_labels: torch.Tensor,
        pos_edge_list: torch.Tensor,
        neg_edge_list: torch.Tensor,
        pos_edge_weights: torch.Tensor,
        neg_edge_weights: torch.Tensor,
        ds_label: torch.Tensor
             ) -> Data:
    
    return Data(x=node_labels,
                edge_index=pos_edge_list,
                neg_edge_index=neg_edge_list,
                edge_attr=pos_edge_weights,
                neg_edge_weights=neg_edge_weights,
                ds_label=ds_label)



class ColocNetDiscreteDataset(InMemoryDataset):
    
    def __init__(self, path: str, name: str,
                 top_k: int, bottom_k: int,
                 ion_labels: torch.Tensor, ds_labels: torch.Tensor, 
                 coloc: Dict[int, pd.DataFrame]) -> None:
        
        self.name = name
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.ion_labels = ion_labels
        self.ds_labels = ds_labels
        self.coloc_dict = coloc
        
        super().__init__(root=path, transform=None, pre_transform=None, pre_filter=None)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name) # type: ignore

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name) # type: ignore

    @property
    def raw_file_names(self) -> str:
        # Data will not be loaded from the disk
        return ''

    @property
    def processed_file_names(self) -> str:
        return f'{self.name}.pt'

    def download(self) -> None:
        # Data will not be downloaded in this class
        pass

    def process(self) -> None:
        data_list = []

        for dsid in self.coloc_dict.keys():

            coloc = self.coloc_dict[dsid]
            # Subset to the correct datasets
            mask = self.ds_labels==dsid
            # Subset ions to the datasets
            masked_ill = self.ion_labels[mask]
            # Convert to numpy
            numpy_ion_labels = np.unique(masked_ill.cpu().detach().numpy())
            # Sort ion labels
            sorted_ion_labels = np.sort(numpy_ion_labels)
            
            # Subset coloc to the correct datasets
            carray = np.array(coloc.loc[sorted_ion_labels, sorted_ion_labels]) # type: ignore

            
            pos_edges: Optional[np.ndarray]=None
            neg_edges: Optional[np.ndarray]=None

            # Loop over all rows of carray
            for i in range(carray.shape[0]):
                # Masking carray to not include the diagonal
                masked_array = carray[i, :]
                masked_array[i] = np.nan

                # Get the top k indices
                top_k_idx = np.argsort(masked_array)[(-self.top_k-1):-1]
                
                # Get the top k ion labels
                # print(pos_edges)
                pos_edges = create_edgelist(pos_edges, i, self.top_k, top_k_idx)


                bottom_k_idx = np.argsort(masked_array)[:self.bottom_k]
                neg_edges = create_edgelist(neg_edges, i, self.bottom_k, bottom_k_idx)

            pos_edge_weights = np.array([carray[x[0], x[1]] for x in pos_edges]) # type: ignore
            neg_edge_weights = np.array([carray[x[0], x[1]] for x in neg_edges]) # type: ignore


            data = get_GraphData(node_labels=torch.tensor(sorted_ion_labels),
                                 pos_edge_list=torch.tensor(pos_edges).transpose(0, 1),
                                 neg_edge_list=torch.tensor(neg_edges).transpose(0, 1),
                                 pos_edge_weights=torch.tensor(pos_edge_weights),
                                 neg_edge_weights=torch.tensor(neg_edge_weights),
                                 ds_label=torch.tensor([dsid]*len(sorted_ion_labels))
                                )
            
            data_list.append(data)


        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{COLOC_NET_DISCRETE_DATA} (hash:{self.name}, len:{len(self)})'
