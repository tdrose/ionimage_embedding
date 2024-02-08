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
import pandas as pd
import seaborn as sns
from typing import Literal

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding.evaluation.scoring import latent_colocinference
from ionimage_embedding.evaluation.gnn import (
    closest_accuracy_aggcoloc_ds, 
    closest_accuracy_latent_ds, 
    closest_accuracy_random_ds,
    mean_coloc_test,
    coloc_umap_ds,
    latent_gnn,
    coloc_ion_labels
    )
from ionimage_embedding.logger import DictLogger




# %%
import os
os.system('nvidia-smi')


# %%
from typing import Dict, Tuple
import numpy as np
from anndata import AnnData
import scanpy as sc
from scipy import sparse
from ionimage_embedding.evaluation.scoring import coloc_knn, most_colocalized
import torch

def coloc_umap_ds_indiv(data: ColocNetData_discrete, k: int=3, 
                        n_components: int=10) -> Dict[int, pd.DataFrame]:

    out = {}
    for i in data._test_set:
        colocs = data.dataset.coloc_dict[i]

        coloc_adata = AnnData(X=colocs)

        # Create nearest neighbor matrix (with n nearest neighbors weighted by coloc)
        coloc_adata.uns['neighbors'] = {'params': {'method': 'umap', 
                                                'metric': 'cosine'},
                                        'connectivities_key': 'connectivities'}

        coloc_adata.obsp['connectivities'] = sparse.csr_array(
            coloc_knn(coloc_adata.X, k=k) # type: ignore
            )
        
        # run scnapy umap function
        sc.tl.umap(coloc_adata, n_components=n_components)

        out[i] = pd.DataFrame(coloc_adata.obsm['X_umap'], index=colocs.index)

    return out

def latent_gnn_indiv(model: gnnDiscrete, data: ColocNetData_discrete) -> Dict[int, pd.DataFrame]:
    out = {}

    for i in data._test_set:
        ## Get data object
        net = data.dataset[i]
        
        # Run network through model
        latent = model.predict(net) # type: ignore

        out[i] = pd.DataFrame(latent.detach().cpu().numpy(), 
                              index=net.x.detach().cpu().numpy()) # type: ignore

    return out

def latent_coloc_indiv(pred: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    out = {}
    for i, df in pred.items():
        out[i] = latent_colocinference(df, torch.tensor(df.index))
    return out

def closest_accuracy_latent_ds_indiv(coloc: Dict[int, pd.DataFrame],
                                     data: ColocNetData_discrete, 
                                     aggcoloc: pd.DataFrame, top: int=3) -> Tuple[float, 
                                                                                  float, float]:
    avail_corr = 0
    avail_total = 1
    trans_corr = 0
    trans_total = 1

    ground_truth = {k: data.dataset.coloc_dict[k] for k in coloc.keys()}

    
    for ds, coloc_df in ground_truth.items():
        
        pred_df = coloc[ds]

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
gnn_layer_type: Literal['GCNConv', 'GATv2Conv', 'GraphConv'] = 'GATv2Conv'
loss_type: Literal['recon', 'coloc'] = 'coloc'
num_layers = 1

RANDOM_NETWORK = False


# %%
# Data
dat = ColocNetData_discrete(KIDNEY_LARGE, test=10, val=5, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
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
                        lightning_device='gpu', loss=loss_type,
                        activation=activation, num_layers=num_layers,
                        gnn_layer_type=gnn_layer_type)


    mylogger = model.train()

    pred_mc, pred_fraction = mean_coloc_test(dat)

    # Predict UMAP for each dataset

    

    pred_cu = coloc_umap_ds_indiv(dat, k=3, n_components=10)
    coloc_cu = latent_coloc_indiv(pred_cu)

    
    pred_gnn_t = latent_gnn_indiv(model, dat)
    coloc_gnn_t = latent_coloc_indiv(pred_gnn_t)
    
    
    acc_l.append(closest_accuracy_aggcoloc_ds(pred_mc, dat, top=top_acc))
    scenario_l.append('Mean coloc')
    eval_l.append('Available')
    frac_l.append(pred_fraction)

    avail, trans, fraction = closest_accuracy_latent_ds_indiv(coloc_cu, dat, pred_mc, top=top_acc)
    scenario_l.append('UMAP')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('UMAP')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    avail, trans, fraction = closest_accuracy_latent_ds_indiv(coloc_gnn_t, dat, pred_mc, top=top_acc)
    scenario_l.append('GNN training')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('GNN training')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    # acc_l.append(closest_accuracy_random_ds(coloc_gnn_t, dat, top=top_acc))
    # scenario_l.append('Random')
    # eval_l.append('Available')
    # frac_l.append(fraction)

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
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(1-df[df['Evaluation']=='Transitivity']['Fraction'].mean()))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'Activation: {activation}')

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%
