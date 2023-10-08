import torch
import torch.nn as nn
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
from .utils import make_symmetric


def run_knn(features: np.ndarray, k: int = 10):
    # Todo: Make a better solution for excluding self neighborhood
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(features)
    _, indices = nbrs.kneighbors(features)
    # Excluding self neighborhood here
    idx = indices[:, 1:]
    adj = np.zeros((features.shape[0], features.shape[0]))
    for i in range(len(idx)):
        adj[i, idx[i]] = 1
    return make_symmetric(adj)


def string_similarity_matrix(string_list):
    n = len(string_list)
    similarity_matrix = [[1 if string_list[i] == string_list[j] else 0 for j in range(n)] for i in range(n)]
    tmp = np.array(similarity_matrix)
    np.fill_diagonal(tmp, 0)
    return tmp


def compute_dataset_ublb(sim_mat, ds_labels: np.ndarray,
                         lower_bound: int, upper_bound: int):
    
    ds_ub = torch.zeros(torch.unique(ds_labels).size(0)).to('cuda')
    ds_lb = torch.zeros(torch.unique(ds_labels).size(0)).to('cuda')
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
    
    return ds_ub, ds_lb


def pseudo_labeling(ub: float, lb: float,
                    sim: torch.tensor,
                    index: np.ndarray,
                    ion_label_mat: np.ndarray,
                    knn: bool, knn_adj: np.ndarray = None,
                    dataset_specific_percentiles: bool = False,
                    dataset_ub: dict = None,
                    dataset_lb: dict = None,
                    ds_labels: np.ndarray = None) -> Tuple[torch.tensor, torch.tensor]:

    if dataset_specific_percentiles:
        ub_m = np.ones(sim.shape)
        # TODO: should be zero matrix for lower bound probably, rethink that!
        lb_m = np.zeros(sim.shape)
        for ds in set(ds_labels):
            ds_v = ds_labels == ds
            ub_m[np.ix_(ds_v, ds_v)] = dataset_ub[ds]
            lb_m[np.ix_(ds_v, ds_v)] = dataset_lb[ds]

        pos_loc = (sim >= ub_m).astype("float64")
        neg_loc = (sim <= lb_m).astype("float64")

    else:
        pos_loc = (sim >= ub).astype("float64")
        neg_loc = (sim <= lb).astype("float64")

    # Align images within KNN
    if knn:
        knn_submat = knn_adj[np.ix_(index, index)]
        # Todo: Not 100% sure with this one, should be checked again
        pos_loc = np.maximum(pos_loc, knn_submat)
        neg_loc = np.minimum(neg_loc, 1-knn_submat)

    # Align the same ions
    ion_submat = ion_label_mat[np.ix_(index, index)]
    pos_loc = np.maximum(pos_loc, ion_submat)
    neg_loc = np.minimum(neg_loc, 1 - ion_submat)

    pos_loc = torch.tensor(pos_loc)
    neg_loc = torch.tensor(neg_loc)

    return pos_loc, neg_loc
