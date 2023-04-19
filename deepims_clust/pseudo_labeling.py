import torch
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
from .utils import make_symmetric


def run_knn(features, k=10):
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


def pseudo_labeling(ub: float, lb: float,
                    sim: torch.tensor,
                    index,
                    ion_label_mat,
                    knn: bool, knn_adj=None) -> Tuple[torch.tensor, torch.tensor]:

    # Align highly similar images
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
