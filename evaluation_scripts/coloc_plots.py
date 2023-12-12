# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import torch
import torch.nn.functional as functional
import torchvision.transforms as T

from ionimage_embedding.models import CRL, CRL2, CRL3
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.crl_data import CRLdata
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

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=100, transformations=T.RandomRotation(degrees=(0, 360)), maxzero=.9)

print(np.isnan(crldat.full_dataset.images).any())
plt.imshow(crldat.full_dataset.images[10])
plt.show()



# %%
colocs = ColocModel(crldat)



# %%
model = CRL3(crldat,
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
            activation='softmax', # softmax
            clip_gradients=None
            )

# %%
device='cuda'
model.train(logger=False)









# ##############
# Visualizations
# ##############

# %% Need better access to loss function:
def contrastive_loss(features, train_datasets, index, train_images, ion_label_mat, device='cuda'):
    
    features = functional.normalize(features, p=2, dim=-1)
    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))
    
    # Compute cosine between all input images
    gt_cosine = torch_cosine(train_images.reshape(train_images.shape[0], -1))
    gt_cosine = gt_cosine.to(device)

    # Only for those values the loss will be evaluated
    ds_mask = torch.zeros(sim_mat.shape, device=device)

    # Loop over all datasets
    for ds in torch.unique(train_datasets):
        ds_v = train_datasets == ds
        
        # Mask to subset similarities just to one dataset
        mask = torch.outer(ds_v, ds_v)
        mask2 = torch.eye(ds_mask.size(0), dtype=torch.bool)
        mask[mask2] = 0.

        # Set maskin with datasets to 1
        ds_mask[mask] = 1

    # Align the same ions
    ion_submat = ion_label_mat[index, :][:, index]
    
    # Set same ions to 1 in target
    ds_mask = torch.maximum(ds_mask, ion_submat)

    return ds_mask

# %%
def plot_ColocHeatmap(ax, data, model, device='cuda'):
    
    mat = torch_cosine(torch.tensor(data.train_dataset.images)).detach().cpu().numpy()
    
    sns.heatmap(mat, vmin=0, vmax=1, ax=ax, cmap='viridis')
    
    #ax.axis('off')
    ax.set_title('Colocalization')

def plot_LossMaskHeatmap(ax, data, model, device='cuda'):
    
    latent = model.inference_embeddings_train(device=device, use_embed_layer=False)
    mask = contrastive_loss(features=torch.tensor(latent, device=device), 
                            train_datasets=data.train_dataset.dataset_labels.to(device),
                            index=torch.tensor(crldat.train_dataset.index, device=device),
                            train_images = torch.tensor(data.train_dataset.images, device=device),
                            ion_label_mat=data.ion_label_mat.to(device), device=device
                            )
    
    sns.heatmap(mask.detach().cpu().numpy(), vmin=0, vmax=1, ax=ax, cmap='viridis')
    
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
