import numpy as np
import torch.nn.functional as functional
import torch
from metaspace import SMInstance
import pandas as pd
import os
import pickle

from ionimage_embedding.dataloader.utils import size_adaption, size_adaption_symmetric

def original_ublb(batch_size, features, uu, ll, train_datasets, index):
    features = functional.normalize(features, p=2, dim=-1)
    features = features / features.norm(dim=1)[:, None]

    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

    sim_numpy = sim_mat.cpu().detach().numpy()

    # Get all sim values from the batch excluding the diagonal
    tmp2 = [sim_numpy[i][j] for i in range(0, batch_size)
            for j in range(batch_size) if i != j]

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


def download_data(evaluation_datasets, testing_dsid, training_dsid):
    training_results = {}
    training_images = {}
    training_if = {}
    polarity = '+'

    sm = SMInstance()

    for k in evaluation_datasets:
        ds = sm.dataset(id=k)
        results = ds.results(database=("HMDB", "v4"), fdr=0.2).reset_index()
        training_results[k] = results
        tmp = ds.all_annotation_images(fdr=0.2, database=("HMDB", "v4"), only_first_isotope=True)
        onsample = dict(zip(results['formula'].str.cat(results['adduct']), ~results['offSample']))
        formula = [x.formula+x.adduct for x in tmp if onsample[x.formula+x.adduct]]
        tmp = np.array([x._images[0] for x in tmp if onsample[x.formula+x.adduct]])
        training_images[k] = tmp
        training_if[k] = formula

    padding_images = size_adaption_symmetric(training_images)

    training_data = []
    training_datasets = [] 
    training_ions = []

    testing_data = []
    testing_datasets = [] 
    testing_ions = []


    for dsid, imgs in padding_images.items():

        if dsid in training_dsid:
            training_data.append(imgs)
            training_datasets += [dsid] * imgs.shape[0]
            training_ions += training_if[dsid]

        testing_data.append(imgs)
        testing_datasets += [dsid] * imgs.shape[0]
        testing_ions += training_if[dsid]


    training_data = np.concatenate(training_data)
    training_datasets = np.array(training_datasets)
    training_ions = np.array(training_ions)

    testing_data = np.concatenate(testing_data)
    testing_datasets = np.array(testing_datasets)
    testing_ions = np.array(testing_ions)
    
    return training_data, training_datasets, training_ions, testing_data, testing_datasets, testing_ions


def load_data(cache=False, cache_folder='/scratch/model_testing'):
    evaluation_datasets = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s',
    '2022-12-07_01h02m53s',
    '2022-12-07_01h01m06s',
    '2022-11-28_22h24m25s',
    '2022-11-28_22h23m30s'
                  ]
    
    training_dsid = evaluation_datasets[:len(evaluation_datasets)-1]
    testing_dsid = evaluation_datasets[len(evaluation_datasets)-1]
    
    if cache:
        # make hash of datasets
        cache_file = 'clr_{}.pickle'.format(''.join(evaluation_datasets))
        
        # Check if cache folder exists
        if not os.path.isdir(cache_folder):
            os.mkdir(cache_folder)
        
        # Download data if it does not exist
        if cache_file not in os.listdir(cache_folder):
            data = download_data(evaluation_datasets, testing_dsid, training_dsid)
            pickle.dump(data, open(os.path.join(cache_folder, cache_file), "wb"))
            print('Saved file: {}'.format(os.path.join(cache_folder, cache_file)))
            return data        
        # Load cached data
        else:
            print('Loading cached data from: {}'.format(os.path.join(cache_folder, cache_file)))
            return pickle.load(open(os.path.join(cache_folder, cache_file), "rb" ) )
    
    else:
        return download_data(evaluation_datasets, testing_dsid, training_dsid)