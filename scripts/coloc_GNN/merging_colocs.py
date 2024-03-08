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

import ionimage_embedding as iie


# %%
import os
os.system('nvidia-smi')

# %% Load data

# Hyperparameters
# #####################
# Hyperparameters
# #####################
# min_images
min_images = 30
# test
test = 5
# val
val = 3
# accuracy top-k
top_acc = 7

RANDOM_NETWORK = False

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



data1 = iie.dataloader.IonImage_data.IonImagedata_random(
    iie.datasets.KIDNEY_LARGE, test=.1, val=.1, transformations=None, fdr=.1,
    min_images=min_images, maxzero=.9, batch_size=10, knn=False,
    colocml_preprocessing=True, cache=True)


# %%
data1.sample_sets()
# %%
import time

start = time.time()
colocs1 = iie.dataloader.get_coloc_model.get_coloc_model(data1)
end = time.time()
print(end - start)

# %%
