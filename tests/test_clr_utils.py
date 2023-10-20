import numpy as np
import torch.nn.functional as functional
import torch


def original_ublb(model, features, uu, ll, train_datasets, index):
    features = functional.normalize(features, p=2, dim=-1)
    features = features / features.norm(dim=1)[:, None]

    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

    sim_numpy = sim_mat.cpu().detach().numpy()

    # Get all sim values from the batch excluding the diagonal
    tmp2 = [sim_numpy[i][j] for i in range(0, model.batch_size)
            for j in range(model.batch_size) if i != j]

    ub = np.percentile(tmp2, uu)
    lb = np.percentile(tmp2, ll)

    return ub, lb, sim_numpy, sim_mat

def original_dataset_ublb(sim_mat: np.ndarray, ds_labels: np.ndarray,
                         lower_bound: int, upper_bound: int):
    ds_ub = {}
    ds_lb = {}
    for dst in torch.unique(ds_labels):
        ds = float(dst.detach().cpu().numpy())
        curr_sim = [sim_mat[i][j] for i in range(0, sim_mat.shape[0])
                    for j in range(sim_mat.shape[0]) if (i != j and ds_labels[i] == ds and ds_labels[j] == ds)]

        if len(curr_sim) > 2:
            ds_ub[ds] = np.percentile(curr_sim, upper_bound)
            ds_lb[ds] = np.percentile(curr_sim, lower_bound)
        else:
            ds_ub[ds] = 1
            ds_lb[ds] = 0

    return ds_ub, ds_lb


def original_ps(ub: float, lb: float,
                sim: torch.tensor,
                index: np.ndarray,
                ion_label_mat: np.ndarray,
                knn: bool, knn_adj: np.ndarray = None,
                dataset_specific_percentiles: bool = False,
                dataset_ub: dict = None,
                dataset_lb: dict = None,
                ds_labels: np.ndarray = None):

    if dataset_specific_percentiles:
        
        ub_m = np.ones(sim.shape)
        # TODO: should be zero matrix for lower bound probably, rethink that!
        lb_m = np.zeros(sim.shape)
        for dst in torch.unique(ds_labels):
            ds = float(dst.detach().cpu().numpy())
            ds_v = ds_labels == ds
            ub_m[np.ix_(ds_v.cpu().numpy(), ds_v.cpu().numpy())] = dataset_ub[ds]
            lb_m[np.ix_(ds_v.cpu().numpy(), ds_v.cpu().numpy())] = dataset_lb[ds]

        pos_loc = (sim >= ub_m).astype("float64")
        neg_loc = (sim <= lb_m).astype("float64")

    else:
        pos_loc = (sim >= ub).astype("float64")
        neg_loc = (sim <= lb).astype("float64")

    # Align images within KNN
    if knn:
        knn_submat = knn_adj[np.ix_(index.cpu().numpy(), index.cpu().numpy())]
        # Todo: Not 100% sure with this one, should be checked again
        pos_loc = np.maximum(pos_loc, knn_submat)
        neg_loc = np.minimum(neg_loc, 1-knn_submat)

    # Align the same ions
    ion_submat = ion_label_mat[np.ix_(index.cpu().numpy(), index.cpu().numpy())]
    pos_loc = np.maximum(pos_loc, ion_submat)
    neg_loc = np.minimum(neg_loc, 1 - ion_submat)

    pos_loc = torch.tensor(pos_loc)
    neg_loc = torch.tensor(neg_loc)

    return pos_loc, neg_loc