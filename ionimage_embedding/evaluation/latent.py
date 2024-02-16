import pandas as pd
import numpy as np
from anndata import AnnData
import scanpy as sc
from scipy import sparse

import torch

from ..coloc.utils import torch_cosine
from ..coloc.coloc import ColocModel
from ._utils import coloc_knn



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


def coloc_umap(colocs: ColocModel, k: int=3, n_components: int=10) -> pd.DataFrame:

    labels = torch.unique(colocs.data.train_dataset.ion_labels)
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