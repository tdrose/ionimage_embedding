# %%

import scanpy as sc
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch
import pickle
from scipy import sparse

from ionimage_embedding.models import CRL, ColocModel
from ionimage_embedding.dataloader import CRLdata
from ionimage_embedding.models.coloc.utils import torch_cosine

# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
except ImportError:
    # Not in IPython, continue with normal Python code
    pass




# %%
# %%
ds_list = [
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

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)


# %%
colocs = ColocModel(crldat)


# %%
def coloc_knn(coloc_matrix: np.ndarray, k: int=30):

    out_matrix = coloc_matrix.copy()

    thresholds = np.sort(out_matrix, axis=1)[:, -k]
    mask = out_matrix < thresholds[:, np.newaxis]

    out_matrix[mask] = 0

    return out_matrix

def coloc_umap(colocs: ColocModel, k: int=3, n_components: int=10) -> pd.DataFrame:

    labels = torch.unique(colocs.data.train_dataset.ion_labels)
    mean_coloc, fraction = colocs.inference(labels)
    mean_coloc[np.isnan(mean_coloc)] = 0
    # Create anndata object

    coloc_adata = sc.AnnData(X=mean_coloc)

    # Create nearest neighbor matrix (with n nearest neighbors weighted by coloc)
    coloc_adata.uns['neighbors'] = {'params': {'method': 'umap', 
                                            'metric': 'cosine'},
                                    'connectivities_key': 'connectivities'}

    coloc_adata.obsp['connectivities'] = sparse.csr_array(
        coloc_knn(coloc_adata.X, k=k)
    )

    # run scnapy umap function
    sc.tl.umap(coloc_adata, n_components=n_components)

    return pd.DataFrame(coloc_adata.obsm['X_umap'], 
                        index=labels.detach().cpu().numpy())

def umap_inference(umap_df: pd.DataFrame, test_labels: torch.Tensor):
    
    sim = pd.DataFrame(torch_cosine(torch.tensor(np.array(umap_df))).detach().cpu().numpy(), 
                       index=umap_df.index, columns=umap_df.index)
    
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

from ionimage_embedding.evaluation.scoring import get_colocs, most_colocalized
def closest_accuracy_umapcoloc(umap_inferred: pd.DataFrame, colocs: ColocModel, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, origin='test')

    pred_df = umap_inferred
    for ds, coloc_df in clc.items():
        if clc[ds].shape[0] > 0:
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

            # create ds coloc df from mean colocs
            curr_cl = np.array(pred_df.loc[clc[ds].index, clc[ds].index]).copy() # type: ignore
            np.fill_diagonal(curr_cl, 0)
            curr_cl[np.isnan(curr_cl)] = 0

            for i in range(len(curr_cl)):

                # Descending sorted most colocalized
                coloc_order = coloc_df.index[np.argsort(curr_cl[i])[::-1]]

                if max_coloc_id[i] in coloc_order[:top]:
                    correct_predictions += 1

                total_predictions += 1

    return correct_predictions / total_predictions



# %%
coloc_embedding = coloc_umap(colocs, k=3, n_components=10)

umap_test_latent = umap_inference(coloc_embedding, colocs.data.test_dataset.ion_labels)

closest_accuracy_umapcoloc(umap_test_latent, colocs, top=5)






# %%
adata = pickle.load(open('/g/alexandr/tim/metaspace_evaluation/230201/single_pixel_adata_Brain_bbknn.pickle', 'rb'))
# %%
