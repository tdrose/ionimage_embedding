# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Dict
from sklearn.metrics import silhouette_score

import torch
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
            num_cluster=50,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.13, # .8
            lower_iteration=0.24, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.18,
            pretraining_epochs=10,
            training_epochs=40, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            cnn_dropout=0.01,
            activation='relu', # softmax
            loss_type='colocContrast', # 'selfContrast', 'colocContrast', 'regContrast',
            resnet='resnet18',
            resnet_pretrained=False,
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
def latent_dataset_silhouette(model: CRL, metric: str='cosine', 
                              dataset: Literal['train', 'val', 'test']='train'):
    if dataset == 'train':
        latent = model.inference_embeddings_train(device=device)
        ds_labels = model.data.train_dataset.dataset_labels.detach().cpu().numpy()
    elif dataset == 'test':
        latent = model.inference_embeddings_test(device=device)
        ds_labels = model.data.test_dataset.dataset_labels.detach().cpu().numpy()
    elif dataset == 'val':
        latent = model.inference_embeddings_val(device=device)
        ds_labels = model.data.test_dataset.dataset_labels.detach().cpu().numpy()
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")
    
    return silhouette_score(X=latent, labels=ds_labels, metric=metric) 

def get_dataset(model: CRL, dataset: Literal['train', 'val', 'test']='train'):
    if dataset == 'train':
        return model.data.train_dataset
    elif dataset == 'test':
        return model.data.test_dataset
    elif dataset == 'val':
        return model.data.val_dataset
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")

def compute_ds_coloc(model: CRL, 
                     dataset: Literal['train', 'val', 'test']='train'):
    if dataset == 'train':
        latent = model.inference_embeddings_train(device=device)
    elif dataset == 'test':
        latent = model.inference_embeddings_test(device=device)
    elif dataset == 'val':
        latent = model.inference_embeddings_val(device=device)
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")
    
    # lcos = torch_cosine(torch.tensor(latent))
    # Make DS specific dict
    out_dict = {}        
    data = get_dataset(model, dataset=dataset)
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

def get_colocs(colocs: ColocModel, dataset: Literal['train', 'val', 'test']='train'):
    if dataset == 'train':
        return colocs.train_coloc
    elif dataset == 'test':
        return colocs.test_coloc
    elif dataset == 'val':
        return colocs.val_coloc
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")
    
def closest_coloc_accuracy_random(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                                  dataset: Literal['train', 'val', 'test']='train'):

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, dataset=dataset)
    for ds, coloc_df in ds_coloc_dict.items():
        
        # Get most colocalized image per dataset
        gt_coloc = np.array(clc[ds]).copy()
        np.fill_diagonal(gt_coloc, 0)
        max_coloc = np.argmax(gt_coloc, axis=0)
        max_coloc_id = clc[ds].index[max_coloc]

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

def closest_coloc_accuracy(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                           dataset: Literal['train', 'val', 'test']='train'):

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, dataset=dataset)
    for ds, coloc_df in ds_coloc_dict.items():
        
        # Get most colocalized image per dataset
        gt_coloc = np.array(clc[ds]).copy()
        np.fill_diagonal(gt_coloc, 0)
        max_coloc = np.argmax(gt_coloc, axis=0)
        max_coloc_id = clc[ds].index[max_coloc]

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

def closest_meancoloc_accuracy(colocs: ColocModel, top: int=5,
                               agg: Literal['mean', 'median']='mean'):

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, dataset='test')

    if agg == 'mean':
        pred_df = colocs.test_mean_coloc
    elif agg == 'median':
        pred_df = colocs.test_median_coloc
    else:
        raise ValueError("`agg` must be one of: ['mean', 'median']")

    for ds, coloc_df in clc.items():
        
        # Get most colocalized image per image per dataset
        gt_coloc = np.array(clc[ds]).copy()
        np.fill_diagonal(gt_coloc, 0)
        max_coloc = np.argmax(gt_coloc, axis=0)
        max_coloc_id = clc[ds].index[max_coloc]

        # create ds coloc df from mean colocs
        curr_cl = np.array(pred_df.loc[clc[ds].index, clc[ds].index]).copy() # type: ignore
        np.fill_diagonal(curr_cl, 0)
        curr_cl[np.isnan(curr_cl)] = 0

        for i in range(len(curr_cl)):

            # Descending sorted most colocalized
            coloc_order = coloc_df.index[np.argsort(curr_cl[i])[::-1]]

            if max_coloc_id[i] in coloc_order[:top]:
                correct_predictions += 1

            total_predictions += 1

    return correct_predictions / total_predictions



    
# %%
ds = 'test'
top = 5
dsc_dict = compute_ds_coloc(model, dataset=ds)
agg='mean'

print('Model accuracy: ', closest_coloc_accuracy(dsc_dict, colocs, top=top, dataset=ds))
print('Random accuracy: ', closest_coloc_accuracy_random(dsc_dict, colocs, top=top, dataset=ds))
if ds == 'test':
    print(f'{agg} accuracy: ', closest_meancoloc_accuracy(colocs, top=top))

print('Silhouette: ', latent_dataset_silhouette(model, dataset=ds))

# %%
