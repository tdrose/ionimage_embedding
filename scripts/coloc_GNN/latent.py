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


import numpy as np
import pickle
import os.path as osp

from ionimage_embedding.datasets import KIDNEY_LARGE
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding.evaluation.scoring import latent_colocinference
from ionimage_embedding.evaluation.gnn import (
    mean_coloc_test,
    latent_gnn,
    coloc_ion_labels,
    closest_accuracy_latent_ds
    )
from ionimage_embedding.coloc.coloc import ColocModel
from ionimage_embedding.dataloader.IonImage_data import IonImagedata_random


# %%
import os
os.system('nvidia-smi')


# %%
# #####################
# Fixed Hyperparameters
# #####################
# device
# min_images
min_images = 30
# test
test = 5
# val
val = 3
# accuracy top-k
top_acc = 3
# Dataset
DSID = KIDNEY_LARGE
# Number of bootstraps
N_BOOTSTRAPS = 100
# Random network
RANDOM_NETWORK = False


# Latent size
hyperparams_avail = {
    'latent_size': 27,
    'top_k': 2,
    'bottom_k':  2,
    'encoding': 'onehot', # 'recon', 'coloc'
    'early_stopping_patience': 2,
    'gnn_layer_type': 'GATv2Conv', # 'GCNConv', 'GATv2Conv', 'GraphConv'
    'loss_type': 'coloc',
    'num_layers': 1,
    'lr': 0.005945,
    'activation': 'none' # 'softmax', 'relu', 'sigmoid', 'none'
}

hyperparams = hyperparams_avail


# %%
# %%
iidata = IonImagedata_random(KIDNEY_LARGE, test=.1, val=.1, transformations=None, fdr=.1,
                             min_images=min_images, maxzero=.9, batch_size=10, 
                             colocml_preprocessing=True, cache=True)


colocs = ColocModel(iidata)

# %%
dat = ColocNetData_discrete(KIDNEY_LARGE, test=test, val=val, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
                    random_network=RANDOM_NETWORK,
                    use_precomputed=True,
                    ds_labels=iidata.full_dataset.dataset_labels,
                    ion_labels=iidata.full_dataset.ion_labels,
                    coloc=colocs.full_coloc,
                    dsl_int_mapper=iidata.dsl_int_mapper,
                    ion_int_mapper=iidata.ion_int_mapper
                    )


# %%
# ###########
# Train model
# ###########

dat.sample_sets()

model = gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                        encoding = hyperparams['encoding'], embedding_dims=40,
                        lr=hyperparams['lr'], training_epochs=130, 
                        early_stopping_patience=hyperparams['early_stopping_patience'],
                        lightning_device='gpu', loss=hyperparams['loss_type'],
                        activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                        gnn_layer_type=hyperparams['gnn_layer_type'])


_ = model.train()

# ##########
# Evaluation
# ##########
pred_mc, _ = mean_coloc_test(dat)

pred_gnn_t = latent_gnn(model, dat, graph='training')
coloc_gnn_t = latent_colocinference(pred_gnn_t, coloc_ion_labels(dat, dat._test_set))

avail, trans, _ = closest_accuracy_latent_ds(coloc_gnn_t, dat, pred_mc, top=top_acc)


# %%

import scanpy as sc
import pandas as pd
# %%

pd.Series(dat.ion_int_mapper)
# %%
adata = sc.AnnData(X=pred_gnn_t, 
                   obs=pd.DataFrame({'ion_label': pd.Series(dat.ion_int_mapper)[pred_gnn_t.index]})
                   )

adata.obs[['molecule', 'ion_label']] = adata.obs['ion_label'].str.split('+', expand=True)
adata.obs[['c', 'rest']] = adata.obs['molecule'].str.split('H', expand=True)

# %%
hmdb_v4 = pd.read_csv('/g/alexandr/tim/metaspace_evaluation/databases/HMDB_v4.csv')
hmdb_v4 = hmdb_v4[['name', 'chemical_formula', 'super_class', 'sub_class', 'class']]

metadata_dict = {'class': [], 'sub_class': [], 'super_class': [], 'names': [], 'ion_label': []}

for i in range(adata.obs.shape[0]):
    mol = adata.obs.iloc[i]['molecule']
    ion = adata.obs.index.values[i]
    tmp = hmdb_v4[hmdb_v4['chemical_formula']==mol]

    if tmp.shape[0] == 0:
        metadata_dict['class'].append('unknown')
        metadata_dict['sub_class'].append('unknown')
        metadata_dict['super_class'].append('unknown')
        metadata_dict['names'].append('unknown')
        metadata_dict['ion_label'].append(ion)
        continue
    
    metadata_dict['class'].append(tmp['class'].values[0])
    metadata_dict['sub_class'].append(tmp['sub_class'].values[0])
    metadata_dict['super_class'].append(tmp['super_class'].values[0])
    
    metadata_dict['names'].append(list(tmp['name']))
    
    metadata_dict['ion_label'].append(ion)


# %%
adata.obs = adata.obs.join(pd.DataFrame(metadata_dict).set_index('ion_label'))

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %%
sc.pl.pca(adata, color='sub_class', dimensions=[0, 1], size=80.)


# %%
sc.pl.umap(adata, color='class', size=80.)

# %%
sc.pl.umap(adata, color='leiden', size=80.)





# %%
import matplotlib.pyplot as plt

LEIDEN_CLUSTER = 9
N_MOLS = 5
N_SAMPLES = 5

ion_labels = iidata.full_dataset.ion_labels

tmp = adata.obs[adata.obs['leiden']==str(LEIDEN_CLUSTER)].reset_index()

# Sample ions
sample_ions = np.random.choice(tmp['ion'].values.astype(int), N_MOLS, replace=False)

# Create figure grid
fig, axs = plt.subplots(N_MOLS, N_SAMPLES, figsize=(5, 5))

for ax in axs.flatten():
    ax.axis('off')

for i, ion in enumerate(sample_ions):
    # Mask the ions
    mask = iidata.full_dataset.ion_labels == ion

    image_idx = np.arange(len(ion_labels))[mask]

    # Shuffle image_idx
    np.random.shuffle(image_idx)

    image_idx = image_idx[:N_SAMPLES]

    for j, idx in enumerate(image_idx):
        axs[i, j].imshow(iidata.full_dataset.images[idx])

fig.suptitle('Leiden cluster: {}'.format(LEIDEN_CLUSTER))

plt.show()
# %%
