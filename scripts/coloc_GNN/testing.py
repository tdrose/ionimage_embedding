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
trans_l = []
avail_l = []

for i in range(N_BOOTSTRAPS):

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
    
    trans_l.append(trans)
    avail_l.append(avail)

res = - (np.mean(trans_l)*2 + np.mean(avail_l))

print(res)

# %%
