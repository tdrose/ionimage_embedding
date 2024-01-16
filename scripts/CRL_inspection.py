# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from typing import Dict
import torch
import pandas as pd

from ionimage_embedding.models import CRL, ColocModel, BioMedCLIP
from ionimage_embedding.models.crl.regContrastModel import regContrastModel
from ionimage_embedding.dataloader import IonImagedata_random
from ionimage_embedding.evaluation.scoring import (
    closest_accuracy_aggcoloc,
    closest_accuracy_latent,
    closest_accuracy_random,
    compute_ds_coloc,
    latent_dataset_silhouette,
    same_ion_similarity,
    coloc_umap,
    latent_colocinference,
    closest_accuracy_coloclatent,
    remove_nan,
    general_mse
)
from ionimage_embedding.evaluation.utils import cluster_latent, latent_centroids_df
from ionimage_embedding.datasets import KIDNEY_SMALL
from ionimage_embedding.evaluation.plots import umap_latent,  umap_allorigin, plot_image_clusters


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


# %%

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')


# %%

crldat = IonImagedata_random(KIDNEY_SMALL, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9, vitb16_compatible=True)

# print(np.isnan(crldat.full_dataset.images).any())
# plt.imshow(crldat.full_dataset.images[10])
# plt.show()

print('Train data:\t\t', crldat.train_dataset.images.shape)
print('Train data:\t\t', crldat.train_dataset.images.shape[0])
print('Validation data:\t', crldat.val_dataset.images.shape[0])
print('Test data:\t\t', crldat.test_dataset.images.shape[0])
# %%
colocs = ColocModel(crldat)



# %%
model = CRL(crldat,
            num_cluster=20,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.0, # .8
            lower_iteration=0.0, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.1, # .18
            pretraining_epochs=10,
            training_epochs=50, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            cnn_dropout=0.01,
            activation='relu', # softmax
            loss_type='regContrast', # 'selfContrast', 'colocContrast', 'regContrast',
            architecture='cnnclust', # 'resnet18
            resnet_pretrained=True,
            clip_gradients=None
            )


# %%
# Initializing the model manually and plotting the gradients and weights
rcm = regContrastModel(
                 height = model._height, 
                 width = model._width,
                 num_cluster = model.num_cluster,
                 ion_label_mat= model.ion_label_mat,
                 knn_adj=model.knn_adj,
                 activation='relu',
                 lr=0.01,
                 architecture = 'resnet18',
                 resnet_pretrained = False,
                 vitb16_pretrained = None,
                 cnn_dropout=0.1, weight_decay=1e-4,
                 clip_gradients= None
                 )

# %%
model_params = rcm.clust.parameters()
optimizer = torch.optim.RMSprop(params=model_params, lr=rcm.lr, weight_decay=rcm.weight_decay)
device='cuda'
rcm = rcm.to(device)

# %%
dl = iter(model.data.get_train_dataloader())

counter = 0

# %%
batch = next(dl)
train_x, index, train_datasets, train_ions, untransformed_images = batch

optimizer.zero_grad()
        
rcm.knn_adj = rcm.knn_adj.to(device)
rcm.ion_label_mat = rcm.ion_label_mat.to(device)

train_datasets = train_datasets.reshape(-1).to(device)
train_ions = train_ions.reshape(-1).to(device)
train_x = train_x.to(device)
index = index.to(device)
untransformed_images = untransformed_images.to(device)

features, x_p = rcm.embed_layers(train_x)

final = rcm.clust.act(features)

loss = rcm.contrastive_loss(features=final, uu=rcm.curr_upper, 
                                ll=rcm.curr_lower, train_datasets=train_datasets, 
                                index=index, train_images=train_x, 
                                raw_images=untransformed_images)

print('Loss: ', loss)

# %%
# Visualize gradient and weights


# %%
loss.backward()
optimizer.step()


counter += 1
if counter == len(dl):
    print('Epoch finished, re-initialize dataloader')
# %%
