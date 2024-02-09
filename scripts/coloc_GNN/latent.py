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
N_BOOTSTRAPS = 30

# Latent size
latent_size = 35
# Top-k
top_k = 2
# Bottom-k
bottom_k = 7
# encoding
encoding = 'onehot'
# early_stopping_patience
early_stopping_patience = 5
# GNN layer type
gnn_layer_type = 'GATv2Conv'
# Loss type
loss_type = 'recon'
# Number of layers
num_layers = 2
# learning rate
lr = 0.0003


# %%
dat = ColocNetData_discrete(KIDNEY_LARGE, test=test, val=val, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=top_k, bottom_k=bottom_k, random_network=False
                    )


# %%
# ###########
# Train model
# ###########


dat.sample_sets()

model = gnnDiscrete(data=dat, 
                    latent_dims=latent_size, 
                    encoding = encoding, # type: ignore
                    embedding_dims=40,
                    lr=lr, 
                    training_epochs=130, 
                    early_stopping_patience=early_stopping_patience,
                    lightning_device='gpu', 
                    loss=loss_type, # type: ignore
                    activation='none', 
                    num_layers=num_layers,
                    gnn_layer_type=gnn_layer_type) # type: ignore


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

# %%
sc.pl.pca(adata, color='sub_class', dimensions=[0, 1], size=80.)

# %%
