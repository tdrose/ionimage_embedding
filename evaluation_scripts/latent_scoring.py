# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score

import torch
import torch.nn.functional as functional
import torchvision.transforms as T

from ionimage_embedding.models import CRL
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
                 fdr=.1, batch_size=100, 
                 transformations=T.RandomRotation(degrees=(0, 360)) , # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

# print(np.isnan(crldat.full_dataset.images).any())
# plt.imshow(crldat.full_dataset.images[10])
# plt.show()



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
            cnn_dropout=0.01,
            activation='relu', # softmax
            loss_type='selfContrast', # 'selfContrast', 'colocContrast', 'regContrast'
            clip_gradients=None
            )

# %%
device='cuda'
mylogger = model.train(logger=True)

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()



# %%
def latent_dataset_silhouette(latent: np.ndarray, ds_labels: np.ndarray, metric: str='cosine'):
    return silhouette_score(X=latent, labels=ds_labels, metric=metric) 
    

latent_dataset_silhouette(latent = model.inference_embeddings_train(),
                          ds_labels = model.data.train_dataset.dataset_labels.detach().cpu().numpy()
                          )


# %%
def compute_ds_coloc(model):

    latent = model.inference_embeddings_train(device=device)
    # lcos = torch_cosine(torch.tensor(latent))
    # Make DS specific dict
    out_dict = {}        
    data = model.data.train_dataset
    # loop over each dataset
    for dsl in torch.unique(data.dataset_labels):
        dsid = int(dsl)
        mask = data.dataset_labels==dsid
        if sum(mask) > 1:
            # Convert data to array (in correct order)
            features = torch.tensor(latent)[mask]
            ions = data.ion_labels[mask]
            ion_sorting_mask = torch.argsort(ions)
            features = features[ion_sorting_mask]
            # Compute coloc matrix (in correct order)
            cos = torch_cosine(features)
            out_dict[dsid] = pd.DataFrame(cos.cpu().detach().numpy(), 
                                            columns=ions[ion_sorting_mask].cpu().detach().numpy(),
                                            index=ions[ion_sorting_mask].cpu().detach().numpy()
                                            )
        else:
            out_dict[dsid] = pd.DataFrame()

    return out_dict

def closest_coloc_accuracy_random(ds_coloc_dict, colocs, top=5, device='cpu'):

    total_predictions = 0
    correct_predictions = 0
    for ds, coloc_df in ds_coloc_dict.items():
        
        # Get most colocalized image per dataset
        gt_coloc = np.array(colocs.train_coloc[ds]).copy()
        np.fill_diagonal(gt_coloc, 0)
        max_coloc = np.argmax(gt_coloc, axis=0)
        max_coloc_id = colocs.train_coloc[ds].index[max_coloc]

        # convert coloc_df to numpy array
        latent_coloc = np.array(coloc_df).copy()
        np.fill_diagonal(latent_coloc, 0)

        for i in range(len(latent_coloc)):

            # Descending sorted most colocalized
            coloc_order = np.random.choice(coloc_df.index, top, replace=False)

            if max_coloc_id[i] in coloc_order:
                correct_predictions += 1

            total_predictions += 1

    return correct_predictions / total_predictions

def closest_coloc_accuracy(ds_coloc_dict, colocs, top=5, device='cpu'):

    total_predictions = 0
    correct_predictions = 0
    for ds, coloc_df in ds_coloc_dict.items():
        
        # Get most colocalized image per dataset
        gt_coloc = np.array(colocs.train_coloc[ds]).copy()
        np.fill_diagonal(gt_coloc, 0)
        max_coloc = np.argmax(gt_coloc, axis=0)
        max_coloc_id = colocs.train_coloc[ds].index[max_coloc]

        # convert coloc_df to numpy array
        latent_coloc = np.array(coloc_df).copy()
        np.fill_diagonal(latent_coloc, 0)

        for i in range(len(latent_coloc)):

            # Descending sorted most colocalized
            coloc_order = coloc_df.index[np.argsort(latent_coloc[i])[::-1]]

            if max_coloc_id[i] in coloc_order[:top]:
                correct_predictions += 1

            total_predictions += 1

    return correct_predictions / total_predictions


    
# %%
dsc_dict = compute_ds_coloc(model)

print(closest_coloc_accuracy(dsc_dict, colocs, top=2))
print(closest_coloc_accuracy_random(dsc_dict, colocs, top=2))

# %%
