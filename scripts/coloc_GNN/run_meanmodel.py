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
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete, MeanColocNetData_discrete
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
dat = MeanColocNetData_discrete(KIDNEY_LARGE, test=10, val=5, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=tk, bottom_k=bk
                    )

mylogger = DictLogger()
# %%
