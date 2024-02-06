
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

from hmac import new
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from typing import Literal, Tuple

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding import ColocModel
from ionimage_embedding.evaluation.scoring import most_colocalized, latent_colocinference
from ionimage_embedding.logger import DictLogger



# %%
import os
os.system('nvidia-smi')


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
import torch
from torch_geometric.data import Data

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
                               data: ColocNetData_discrete,
                               aggcoloc: pd.DataFrame, 
                               top: int=5) -> Tuple[float, float, float]:

    avail_corr = 0
    avail_total = 1
    trans_corr = 0
    trans_total = 1

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
# Hyperparameters
scenario_l = []
acc_l = []
eval_l = []
frac_l = []
tk = 5 # best for reconstruction loss: 2
bk = 5 # best for reconstruction loss: 1
min_images = tk + bk + 1 # Default 6
top_acc = 3
encoding = 'onehot'
early_stopping_patience = 3
activation: Literal['softmax', 'relu', 'sigmoid', 'none'] = 'none'
num_layers = 3

RANDOM_NETWORK = False

# %%
# Data
dat = ColocNetData_discrete(KIDNEY_LARGE, test=10, val=5, 
                    cache_images=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=tk, bottom_k=bk, random_network=RANDOM_NETWORK
                    )

mylogger = DictLogger()

# %%
# Training
for i in range(100):
    print('# #######')
    print(f'# Iteration {i}')
    print('# #######')
    
    dat.sample_sets()

    model = gnnDiscrete(data=dat, latent_dims=20, 
                        encoding = encoding, embedding_dims=10,
                        lr=1e-3, training_epochs=130, 
                        early_stopping_patience=early_stopping_patience,
                        lightning_device='gpu', loss='coloc',
                        activation=activation, num_layers=num_layers)


    mylogger = model.train()

    pred_mc, pred_fraction = mean_coloc_test(dat)
    

    pred_cu = coloc_umap_ds(dat, k=3, n_components=10)
    

    coloc_cu = latent_colocinference(pred_cu, coloc_ion_labels(dat, dat._test_set))

    
    pred_gnn_t = latent_gnn(model, dat, graph='training')
    coloc_gnn_t = latent_colocinference(pred_gnn_t, coloc_ion_labels(dat, dat._test_set))
    
    
    acc_l.append(closest_accuracy_aggcoloc_ds(pred_mc, dat, top=top_acc))
    scenario_l.append('Mean coloc')
    eval_l.append('Available')
    frac_l.append(pred_fraction)

    avail, trans, fraction = closest_accuracy_latent_ds(coloc_cu, dat, pred_mc, top=top_acc)
    scenario_l.append('UMAP')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('UMAP')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    avail, trans, fraction = closest_accuracy_latent_ds(coloc_gnn_t, dat, pred_mc, top=top_acc)
    scenario_l.append('GNN training')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('GNN training')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    acc_l.append(closest_accuracy_random_ds(coloc_gnn_t, dat, top=top_acc))
    scenario_l.append('Random')
    eval_l.append('Available')
    frac_l.append(fraction)
    

    #print(f'Fraction of predicted colocs: {pred_fraction:.2f}')

# %%
# Evaluation
df = pd.DataFrame({'Scenario': scenario_l, 'Accuracy': acc_l, 
                   'Evaluation': eval_l, 'Fraction': frac_l})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Available'], x='Scenario', y='Accuracy', ax=ax1)
ax1.set_title(f'KIDNEY_LARGE (top-k: {tk}, bottom-k: {bk}, encoding: {encoding})')
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Available)')
ax1.set_ylim(0, 1)

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Scenario', y='Accuracy', ax=ax2)
ax2.set_title('Mean transifivity fraction: {:.2f}'.format(1-df[df['Evaluation']=='Transitivity']['Fraction'].mean()))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'Activation: {activation}')

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%
