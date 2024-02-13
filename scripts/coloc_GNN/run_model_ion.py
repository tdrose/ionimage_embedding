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

from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.dataloader.IonImage_data import IonImagedata_random
from ionimage_embedding.coloc.coloc import ColocModel

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
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
import ionimage_embedding.evaluation.scoring as scoring

from ionimage_embedding.logger import DictLogger




# %%
import os
os.system('nvidia-smi')


# %%
# Hyperparameters
# #####################
# Hyperparameters
# #####################
# min_images
min_images = 11
# test
test = 1
# val
val = 1
# accuracy top-k
top_acc = 3
# Dataset
DSID = KIDNEY_LARGE
# Number of bootstraps
N_BOOTSTRAPS = 100


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

hyperparams_transitivity = {
    'latent_size': 23,
    'top_k': 3,
    'bottom_k':  7,
    'encoding': 'onehot', # 'recon', 'coloc'
    'early_stopping_patience': 10,
    'gnn_layer_type': 'GCNConv', # 'GCNConv', 'GATv2Conv', 'GraphConv'
    'loss_type': 'coloc',
    'num_layers': 1,
    'lr': 0.009140,
    'activation': 'none' # 'softmax', 'relu', 'sigmoid', 'none'
}


hyperparams = hyperparams_transitivity


scenario_l = []
acc_l = []

RANDOM_NETWORK = False

# %% II Data
iidata = IonImagedata_random(KIDNEY_LARGE, test=.1, val=.1, transformations=None, fdr=.1,
                             min_images=min_images, maxzero=.9, batch_size=10, 
                             colocml_preprocessing=True, cache=True)


colocs = ColocModel(iidata)

# %% ColocNet Data

dat = ColocNetData_discrete([""], 
                    test=test, val=val, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
                    random_network=RANDOM_NETWORK,
                    use_precomputed=True,
                    ds_labels=iidata.train_dataset.dataset_labels,
                    ion_labels=iidata.train_dataset.ion_labels,
                    coloc=colocs.train_coloc,
                    dsl_int_mapper=iidata.dsl_int_mapper,
                    ion_int_mapper=iidata.ion_int_mapper,
                    force_reload=True,
                    )

mylogger = DictLogger()


# %%
model = gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                        encoding = hyperparams['encoding'], embedding_dims=40,
                        lr=hyperparams['lr'], training_epochs=130, 
                        early_stopping_patience=hyperparams['early_stopping_patience'],
                        lightning_device='gpu', loss=hyperparams['loss_type'],
                        activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                        gnn_layer_type=hyperparams['gnn_layer_type'])

# %%
mylogger = model.train()


# %%

pred_mc, pred_fraction = mean_coloc_test(dat)


coloc_embedding = scoring.coloc_umap(colocs, k=3, n_components=5)
umap_coloc_inferred = latent_colocinference(coloc_embedding, colocs.data.test_dataset.ion_labels)
perf = scoring.closest_accuracy_latent(umap_coloc_inferred, colocs, top=top_acc)
scenario_l.append('UMAP')
acc_l.append(perf)

perf = scoring.closest_accuracy_aggcoloc(colocs, top=top_acc)
scenario_l.append('Mean coloc')
acc_l.append(perf)


pred_gnn_t = latent_gnn(model, dat, graph='training')
perf = None
scenario_l.append('GNN training')
acc_l.append(perf)

coloc_gnn_t = latent_colocinference(pred_gnn_t, coloc_ion_labels(dat, dat._test_set))
