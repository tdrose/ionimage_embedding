# This file contains code from the following repository:
# https://github.com/tdrose/deep_mzClustering, 2023, MIT License
# Specifically from the file: mz_clustering/pseudo_labeling.py (commit: 8a4ffcb)

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import numpy as np




def string_similarity_matrix(string_list):
    n = len(string_list)
    similarity_matrix = [[1 if string_list[i] == string_list[j] else 0 
                          for j in range(n)] for i in range(n)]
    tmp = np.array(similarity_matrix)
    np.fill_diagonal(tmp, 0)
    return tmp


def compute_dataset_ublb(sim_mat, ds_labels,
                         lower_bound: int, 
                         upper_bound: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    ds_ub = torch.zeros(int(torch.max(torch.unique(ds_labels)))+1, device=device)
    ds_lb = torch.zeros(int(torch.max(torch.unique(ds_labels)))+1, device=device)
    for ds in torch.unique(ds_labels):
        
        labels = ds_labels==ds
        if labels.sum() > 2:
            ds_mat = sim_mat[labels, :][:, labels]
            mask = torch.eye(ds_mat.size(0), dtype=torch.bool)
            masked_dsmat = ds_mat[~mask]
            ds_ub[ds] = torch.quantile(masked_dsmat, upper_bound/100)
            ds_lb[ds] = torch.quantile(masked_dsmat, lower_bound/100)
            
        else:
            # TODO: Potential issue here with numbers not being on the gpu
            ds_ub[ds] = 1
            ds_lb[ds] = 0
    
    return ds_ub.detach(), ds_lb.detach()


def pseudo_labeling(ub: Union[float, torch.Tensor], lb: Union[float, torch.Tensor],
                    sim: torch.Tensor,
                    index: np.ndarray,
                    ion_label_mat: torch.Tensor,
                    knn: bool, knn_adj: Optional[torch.Tensor],
                    dataset_ub: torch.Tensor,
                    dataset_lb: torch.Tensor,
                    dataset_specific_percentiles: bool = False,
                    ds_labels: Optional[np.ndarray] = None, 
                    device = None) -> Tuple[torch.Tensor, torch.Tensor]:

    if dataset_specific_percentiles:

        ub_m = torch.ones(sim.shape, device=device)#.to(device)
        # TODO: should be zero matrix for lower bound probably, rethink that!
        lb_m = torch.zeros(sim.shape, device=device)#.to(device)
        for ds in torch.unique(ds_labels):
            ds_v = ds_labels == ds
            mask = torch.outer(ds_v, ds_v)
            
            ub_m[mask] = dataset_ub[ds]
            lb_m[mask] = dataset_lb[ds]

        pos_loc = (sim >= ub_m).float()
        neg_loc = (sim <= lb_m).float()

    else:
        pos_loc = (sim >= ub).float()
        neg_loc = (sim <= lb).float()

    # Align images within KNN
    if knn:
        if knn_adj is None:
            raise ValueError('KNN adjacency matrix is None, but knn is True.' 
                             'Make sure to compute KNN when loading the data.')
        knn_submat = knn_adj[index, :][:, index]
        # Todo: Not 100% sure with this one, should be checked again
        pos_loc = torch.maximum(pos_loc, knn_submat)
        neg_loc = torch.minimum(neg_loc, 1-knn_submat)

    # Align the same ions
    ion_submat = ion_label_mat[index, :][:, index]
    
    pos_loc = torch.maximum(pos_loc, ion_submat)
    neg_loc = torch.minimum(neg_loc, 1 - ion_submat)

    return pos_loc, neg_loc
