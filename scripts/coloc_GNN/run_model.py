
# %%
# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
    print('Running in IPython, auto-reload enabled!')
except ImportError:
    # Not in IPython, continue with normal Python code
    pass

import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding import ColocModel
from ionimage_embedding.evaluation.scoring import most_colocalized, latent_colocinference



# %%
import os
os.system('nvidia-smi')


# %%
dat = ColocNetData_discrete(KIDNEY_SMALL, test=2, val=1, 
                 cache_images=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=1, min_images=6, top_k=3,
                 maxzero=.9)



# %%
model = gnnDiscrete(data=dat, latent_dims=10, 
                    encoding = 'learned', embedding_dims=10,
                    lr=1e-3, training_epochs=50, lightning_device='gpu')


# %%
mylogger = model.train()

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()



# %%
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
    
    train_ds = data._train_set
    
    test_ions = coloc_ion_labels(data, prediction_set)
    # Compute mean coloc across training datasets
    mean_colocs = ColocModel._inference(ds_colocs={k: v for k, v in data.dataset.coloc_dict.items() if k in train_ds}, 
                                        ion_labels=test_ions, agg='mean')
    
    return mean_colocs

import pandas as pd
from anndata import AnnData
import scanpy as sc
from scipy import sparse
from ionimage_embedding.evaluation.scoring import coloc_knn

def coloc_umap_ds(data: ColocNetData_discrete, k: int=3, n_components: int=10) -> pd.DataFrame:

    mean_coloc, _ = mean_coloc_train(data)
    mean_coloc[np.isnan(mean_coloc)] = 0
    
    labels = coloc_ion_labels(data, data._train_set)

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

def latent_gnn(model: gnnDiscrete, data: ColocNetData_discrete) -> pd.DataFrame:
    training_data = data.dataset.index_select(data._train_set)

    return model.predict_centroids_df(training_data) # type: ignore
    

# %%
# Scoring
def closest_accuracy_aggcoloc_ds(predictions: pd.DataFrame, 
                                 data: ColocNetData_discrete, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0

    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data._test_set}

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
                               data: ColocNetData_discrete, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0
    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data._test_set}

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

                if max_coloc_id[i] in coloc_order[:top]:
                    correct_predictions += 1

                total_predictions += 1

    return correct_predictions / total_predictions

def closest_accuracy_random_ds(latent: pd.DataFrame, 
                               data: ColocNetData_discrete, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0
    ground_truth = {k: v for k, v in data.dataset.coloc_dict.items() if k in data._test_set}

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


# %%
scenario_l = []
acc_l = []

for i in range(10):
    
    dat = ColocNetData_discrete(KIDNEY_LARGE, test=2, val=1, 
                    cache_images=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=6, top_k=3,
                    maxzero=.9)

    # model = gnnDiscrete(data=dat, latent_dims=10, 
    #                     encoding = 'onehot', embedding_dims=10
    #                     lr=1e-3, training_epochs=170, lightning_device='gpu')
    model = gnnDiscrete(data=dat, latent_dims=10, 
                    encoding = 'learned', embedding_dims=10,
                    lr=1e-3, training_epochs=40, lightning_device='gpu')


    mylogger = model.train()

    pred_mc, pred_fraction = mean_coloc_test(dat)
    

    pred_cu = coloc_umap_ds(dat, k=3, n_components=10)
    pred_gnn = latent_gnn(model, dat)

    coloc_cu = latent_colocinference(pred_cu, coloc_ion_labels(dat, dat._test_set))
    coloc_gnn = latent_colocinference(pred_gnn, coloc_ion_labels(dat, dat._test_set))

    top = 3

    
    scenario_l.append('Mean coloc')
    acc_l.append(closest_accuracy_aggcoloc_ds(pred_mc, dat, top=top))
    scenario_l.append('UMAP')
    acc_l.append(closest_accuracy_latent_ds(coloc_cu, dat, top=top))
    scenario_l.append('GNN')
    acc_l.append(closest_accuracy_latent_ds(coloc_gnn, dat, top=top))
    scenario_l.append('Random')
    acc_l.append(closest_accuracy_random_ds(coloc_gnn, dat, top=top))

    #print(f'Fraction of predicted colocs: {pred_fraction:.2f}')

# %%
df = pd.DataFrame({'Scenario': scenario_l, 'Accuracy': acc_l})
sns.boxplot(data=df, x='Scenario', y='Accuracy')

# %%
