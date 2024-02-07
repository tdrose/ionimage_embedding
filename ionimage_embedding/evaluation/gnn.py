from typing import Literal, Tuple
import pandas as pd
from anndata import AnnData
import scanpy as sc
from scipy import sparse
import numpy as np

import torch
from torch_geometric.data import Data

from ..dataloader.ColocNet_data import ColocNetData_discrete
from ..coloc.coloc import ColocModel
from ..models import gnnDiscrete
from .scoring import most_colocalized, coloc_knn


def mean_coloc_test(data: ColocNetData_discrete):

    return mean_colocs(data, data._test_set)

def mean_coloc_train(data: ColocNetData_discrete):
    
    return mean_colocs(data, data._train_set)


def coloc_ion_labels(data: ColocNetData_discrete, prediction_set: np.ndarray):
    test_ions = []
    for dsid in prediction_set:
        test_ions.extend(data.dataset.coloc_dict[dsid].columns)
    test_ions = torch.tensor(list(set(test_ions)))
    return test_ions
    

def mean_colocs(data: ColocNetData_discrete, prediction_set: np.ndarray):
    
    train_data = {k: v for k, v in data.dataset.coloc_dict.items() if k in data.get_train_dsids()}

    train_ds = data._train_set
    
    test_ions = coloc_ion_labels(data, prediction_set)
    # Compute mean coloc across training datasets
    mean_colocs = ColocModel._inference(ds_colocs=train_data, 
                                        ion_labels=test_ions, agg='mean')
    
    return mean_colocs

def coloc_umap_ds(data: ColocNetData_discrete, k: int=3, n_components: int=10) -> pd.DataFrame:

    mean_coloc, _ = mean_coloc_train(data)
    mean_coloc[np.isnan(mean_coloc)] = 0
    
    labels = coloc_ion_labels(data, data.get_train_dsids().detach().cpu().numpy())

    # Create anndata object
    coloc_adata = AnnData(X=mean_coloc)

    # Create nearest neighbor matrix (with n nearest neighbors weighted by coloc)
    coloc_adata.uns['neighbors'] = {'params': {'method': 'umap', 
                                            'metric': 'cosine'},
                                    'connectivities_key': 'connectivities'}

    coloc_adata.obsp['connectivities'] = sparse.csr_array(
        coloc_knn(coloc_adata.X, k=k) # type: ignore
    )

    # run scnapy umap function
    sc.tl.umap(coloc_adata, n_components=n_components)

    return pd.DataFrame(coloc_adata.obsm['X_umap'], 
                        index=labels.detach().cpu().numpy())

def latent_gnn(model: gnnDiscrete, data: ColocNetData_discrete, 
               graph: Literal['training', 'unconnected', 'union']='training') -> pd.DataFrame:
    training_data = data.dataset.index_select(data._train_set)

    if graph == 'training':
        return model.predict_centroids_df(training_data) # type: ignore
    elif graph == 'unconnected':
        return model.predict_from_unconnected(training_data) # type: ignore
    elif graph == 'union':
        return model.predict_from_union(training_data) # type: ignore
    else:
        raise ValueError('graph must be either "training", "unconnected", or "union"')

# %%
# Scoring
def closest_accuracy_aggcoloc_ds(predictions: pd.DataFrame, 
                                 data: ColocNetData_discrete, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0

    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data.get_test_dsids()}

    for ds, coloc_df in ground_truth.items():
        if coloc_df.shape[0] > 0:
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=ground_truth, ds=ds)

            # create ds coloc df from mean colocs
            curr_cl = np.array(predictions.loc[ground_truth[ds].index, 
                                               ground_truth[ds].index]).copy() # type: ignore
            
            np.fill_diagonal(curr_cl, 0)
            curr_cl[np.isnan(curr_cl)] = 0

            for i in range(len(curr_cl)):
                
                # Descending sorted most colocalized
                coloc_order = coloc_df.index[np.argsort(curr_cl[i])[::-1]]

                if max_coloc_id[i] in coloc_order[:top]:
                    correct_predictions += 1

                total_predictions += 1

    return correct_predictions / total_predictions

def closest_accuracy_latent_ds(latent: pd.DataFrame, 
                               data: ColocNetData_discrete,
                               aggcoloc: pd.DataFrame, 
                               top: int=5) -> Tuple[float, float, float]:

    avail_corr = 0
    avail_total = 1
    trans_corr = 0
    trans_total = 1

    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data.get_test_dsids()}

    pred_df = latent
    for ds, coloc_df in ground_truth.items():
        if ground_truth[ds].shape[0] > 0:
            
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=ground_truth, ds=ds)

            # create dataset specific coloc df from mean colocs
            curr_cl = np.array(pred_df.loc[ground_truth[ds].index, 
                                           ground_truth[ds].index]).copy() # type: ignore
            
            # Set diagonal to zero (no self coloc)
            np.fill_diagonal(curr_cl, 0)
            
            # Set NA values to zero
            curr_cl[np.isnan(curr_cl)] = 0

            # Loop over each molecule
            for i in range(len(curr_cl)):

                # Descending sorted most colocalized
                mask = np.argsort(curr_cl[i])[::-1]

                # We are using the coloc_df index for the curr_cl array
                coloc_order = coloc_df.index[mask]

                if np.isnan(aggcoloc.loc[ground_truth[ds].index[i], max_coloc_id[i]]):
                    if max_coloc_id[i] in coloc_order[:top]:
                        trans_corr += 1

                    trans_total += 1
                else:
                    if max_coloc_id[i] in coloc_order[:top]:
                        avail_corr += 1

                    avail_total += 1

    return avail_corr / avail_total, trans_corr/trans_total, avail_total/(trans_total+avail_total)

def closest_accuracy_random_ds(latent: pd.DataFrame, 
                               data: ColocNetData_discrete, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0
    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data.get_test_dsids()}

    pred_df = latent
    for ds, coloc_df in ground_truth.items():
        if ground_truth[ds].shape[0] > 0:
            
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=ground_truth, ds=ds)

            # create dataset specific coloc df from mean colocs
            curr_cl = np.array(pred_df.loc[ground_truth[ds].index, 
                                           ground_truth[ds].index]).copy() # type: ignore
            
            # Set diagonal to zero (no self coloc)
            np.fill_diagonal(curr_cl, 0)
            
            # Set NA values to zero
            curr_cl[np.isnan(curr_cl)] = 0

            # Loop over each molecule
            for i in range(len(curr_cl)):

                if max_coloc_id[i] in np.random.choice(np.arange(len(curr_cl[i])), 
                                                       size=top, replace=False):
                    correct_predictions += 1

                total_predictions += 1

    return correct_predictions / total_predictions
