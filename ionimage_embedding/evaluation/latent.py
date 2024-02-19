from typing import Literal, Union
import pandas as pd
import numpy as np
from anndata import AnnData
import scanpy as sc
from scipy import sparse

import torch

from ..coloc.utils import torch_cosine
from ..coloc.coloc import ColocModel
from ..dataloader.ColocNet_data import ColocNetData_discrete
from ..models import gnnDiscrete, CRL, BioMedCLIP, CVAE
from ._utils import coloc_knn
from .utils_iid import latent_centroids
from .utils_gnn import mean_coloc_train, coloc_ion_labels

def coloc_umap_iid(colocs: ColocModel, k: int=3, n_components: int=10) -> pd.DataFrame:

    labels = torch.unique(colocs.train_dataset.ion_labels)
    mean_coloc, fraction = colocs.inference(labels)
    mean_coloc[np.isnan(mean_coloc)] = 0
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


def coloc_umap_gnn(data: ColocNetData_discrete, k: int=3, n_components: int=10) -> pd.DataFrame:

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


def latent_colocinference(latent_df: pd.DataFrame, test_labels: torch.Tensor):
    
    sim = pd.DataFrame(torch_cosine(torch.tensor(np.array(latent_df))).detach().cpu().numpy(), 
                       index=latent_df.index, columns=latent_df.index)
    
    sorted_ion_labels = np.sort(np.unique(test_labels.detach().cpu().numpy()))
    out = np.zeros((len(sorted_ion_labels), len(sorted_ion_labels)))
    
    for i1 in range(len(sorted_ion_labels)):
            for i2 in range(i1, len(sorted_ion_labels)):
                if i1 == i2:
                    out[i1, i2] = np.nan
                elif sorted_ion_labels[i1] == sorted_ion_labels[i2]:
                    out[i1, i2] = np.nan
                    out[i2, i1] = np.nan
                else:
                    if sorted_ion_labels[i1] in sim.index and sorted_ion_labels[i2] in sim.index:
                        out[i1, i2] = sim.loc[sorted_ion_labels[i1], sorted_ion_labels[i2]]
                        out[i2, i1] = sim.loc[sorted_ion_labels[i1], sorted_ion_labels[i2]]
                    else:
                        out[i1, i2] = np.nan
                        out[i2, i1] = np.nan
    return pd.DataFrame(out, index=sorted_ion_labels, columns=sorted_ion_labels)



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



def latent_iid(model: Union[CRL, BioMedCLIP, CVAE], 
               origin: Literal['train', 'val', 'test']='train') -> pd.DataFrame:
    # Call the latent_centroids function and create a dataframe of the results 
    # by using the centroid labels as the index
    centroids, labels = latent_centroids(model=model, origin=origin)
    df = pd.DataFrame(centroids, index=labels)
    df.index.name = 'ion'
    # sort the dataframe by the index
    df = df.sort_index(inplace=False)

    return df
