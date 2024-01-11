# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import torch
import torch.nn.functional as functional
import torchvision.transforms as T

from ionimage_embedding.models import CRL
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.IonImage_data import IonImagedata_random
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

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')


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

crldat = IonImagedata_random(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=100, transformations=T.RandomRotation(degrees=(0, 360)), maxzero=.9)

print(np.isnan(crldat.full_dataset.images).any())
plt.imshow(crldat.full_dataset.images[10])
plt.show()



# %%
colocs = ColocModel(crldat)



# %%
model = CRL(crldat,
            num_cluster=30,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.13, # .8
            lower_iteration=0.24, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.18,
            pretraining_epochs=10,
            training_epochs=20, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            activation='relu', # softmax
            loss_type='regContrast',
            clip_gradients=None
            )

# %%
device='cuda'
model.train(logger=False)









# ##############
# Visualizations
# ##############

# %%
def plot_ColocHeatmap(ax, data, model, device='cuda'):
    
    mat = torch_cosine(torch.tensor(data.train_dataset.images)).detach().cpu().numpy()
    
    sns.heatmap(mat, vmin=0, vmax=1, ax=ax, cmap='viridis')
    
    #ax.axis('off')
    ax.set_title('Colocalization')

def plot_LossMaskHeatmap(ax, data, model, device='cuda'):
    
    latent = model.inference_embeddings_train(device=device, use_embed_layer=False)
    
    mask = model.get_loss_mask(latent=torch.tensor(latent, device=device), 
                               index=torch.tensor(crldat.train_dataset.index, device=device), 
                               train_datasets=data.train_dataset.dataset_labels.to(device), 
                               train_images=torch.tensor(data.train_dataset.images, device=device), 
                               uu=90, ll=10, device=device)
    if np.min(mask < 0):
        sns.heatmap(mask, vmin=-1, vmax=1, ax=ax, cmap='RdBu')
    else:
        sns.heatmap(mask, vmin=0, vmax=1, ax=ax, cmap='viridis')
    
    ax.axis('off')
    ax.set_title('Loss Mask')

def plot_latentHeatmap(ax, data, model, device='cuda'):

    latent = model.inference_embeddings_train(device=device, use_embed_layer=False)
    
    mat = torch_cosine(torch.tensor(latent)).detach().cpu().numpy()
    
    sns.heatmap(mat, vmin=0, vmax=1, ax=ax, cmap='viridis')
    
    ax.axis('off')
    ax.set_title('Latent space cosine')




# %%
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(21, 7))

plot_ColocHeatmap(ax0, crldat, model, device=device)
plot_LossMaskHeatmap(ax1, crldat, model, device=device)
plot_latentHeatmap(ax2, crldat, model, device=device)

plt.show()


# %% Latent space UMAPs


reducer = umap.UMAP()
embeddings = model.inference_embeddings_train(device=device, use_embed_layer=False)

data = pd.DataFrame(reducer.fit_transform(embeddings))
data['dataset'] = crldat.train_dataset.dataset_labels.cpu().numpy().astype(str)

sns.scatterplot(data, x=0, y=1, hue='dataset').set(title='Training data')


# %%
def compute_centroids(embeddings, ion_labels):
    
    ion_centroids = []
    centroid_labels = []
    for i in set(ion_labels):
        if len(embeddings[ion_labels==i]) > 1:
            a = embeddings[ion_labels==i]
            ion_centroids.append(a.sum(axis=0)/a.shape[0])
            centroid_labels.append(i)
        else:
            ion_centroids.append(embeddings[ion_labels==i][0])
            centroid_labels.append(i)
            
    return np.stack(ion_centroids), centroid_labels

centroids, centroid_labels = compute_centroids(embeddings, crldat.train_dataset.ion_labels.cpu().numpy())
data = pd.DataFrame(reducer.fit_transform(centroids))

sns.scatterplot(data, x=0, y=1).set(title='Training data')


# %%
