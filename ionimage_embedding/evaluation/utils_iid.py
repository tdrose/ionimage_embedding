from typing import Literal, Union, Tuple, Dict
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn.cluster as cluster
from sklearn.metrics import pairwise_kernels

import torch

from ..coloc.coloc import ColocModel
from ..coloc.utils import torch_cosine
from ..models import CRL, BioMedCLIP, CVAE
from ..torch_datasets.mzImageDataset import mzImageDataset
from ._utils import compute_umap


def get_colocs(colocs: ColocModel, origin: Literal['train', 'val', 'test']='train'):
    if origin == 'train':
        return colocs.train_coloc
    elif origin == 'test':
        return colocs.test_coloc
    elif origin == 'val':
        return colocs.val_coloc
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")

def get_mzimage_dataset(model: Union[CRL, BioMedCLIP, CVAE], 
                        origin: Literal['train', 'val', 'test']='train') -> mzImageDataset:
    if origin == 'train':
        return model.data.train_dataset
    elif origin == 'test':
        return model.data.test_dataset
    elif origin == 'val':
        return model.data.val_dataset
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")

def get_latent(model: Union[CRL, BioMedCLIP, CVAE], device: str='cpu',
               origin: Literal['train', 'val', 'test']='train') -> np.ndarray:
    if origin == 'train':
        latent = model.inference_embeddings_train(device=device)
    elif origin == 'test':
        latent = model.inference_embeddings_test(device=device)
    elif origin == 'val':
        latent = model.inference_embeddings_val(device=device)
    else:
        raise ValueError("`origin` must be one of: ['train', 'val', 'test']")
    
    return latent

def get_ds_labels(model: Union[CRL, BioMedCLIP, CVAE],
                  origin: Literal['train', 'val', 'test']='train') -> np.ndarray:
    if origin == 'train':
        ds_labels = model.data.train_dataset.dataset_labels.detach().cpu().numpy()
    elif origin == 'test':
        ds_labels = model.data.test_dataset.dataset_labels.detach().cpu().numpy()
    elif origin == 'val':
        ds_labels = model.data.val_dataset.dataset_labels.detach().cpu().numpy()
    else:
        raise ValueError("`origin` must be one of: ['train', 'val', 'test']")
    
    return ds_labels

def get_ion_labels(model: Union[CRL, BioMedCLIP, CVAE],
                   origin: Literal['train', 'val', 'test']='train') -> np.ndarray:
    if origin == 'train':
        ion_labels = model.data.train_dataset.ion_labels.detach().cpu().numpy()
    elif origin == 'test':
        ion_labels = model.data.test_dataset.ion_labels.detach().cpu().numpy()
    elif origin == 'val':
        ion_labels = model.data.val_dataset.ion_labels.detach().cpu().numpy()
    else:
        raise ValueError("`origin` must be one of: ['train', 'val', 'test']")
    
    return ion_labels

def get_ion_images(model: Union[CRL, BioMedCLIP, CVAE],
                   origin: Literal['train', 'val', 'test']='train') -> np.ndarray:
    if origin == 'train':
        images = model.data.train_dataset.images
    elif origin == 'test':
        images = model.data.test_dataset.images
    elif origin == 'val':
        images = model.data.val_dataset.images
    else:
        raise ValueError("`origin` must be one of: ['train', 'val', 'test']")

    return images

def latent_centroids(model: Union[CRL, BioMedCLIP, CVAE], 
                     origin: Literal['train', 'val', 'test']='train') -> Tuple[np.ndarray, 
                                                                               np.ndarray]:

    latent = get_latent(model=model, origin=origin)
    ion_labels = get_ion_labels(model=model, origin=origin)

    ion_centroids = []
    centroid_labels = []
    for i in set(ion_labels):
        if len(latent[ion_labels==i]) > 1:
            a = latent[ion_labels==i]
            ion_centroids.append(np.mean(a, axis=0))
            centroid_labels.append(i)
        else:
            ion_centroids.append(latent[ion_labels==i][0])
            centroid_labels.append(i)
            
    return np.stack(ion_centroids), np.array(centroid_labels)

def cluster_latent(model: Union[CRL, BioMedCLIP, CVAE], n_clusters: int=10, plot: bool=False, device='cpu',
                   origin: Literal['train', 'val', 'test']='train'):
    
    latent = get_latent(model=model, device=device, origin=origin)
    umap_df = compute_umap(latent)

    ds_labels = get_ds_labels(model=model, origin=origin)
    ds_labels = ds_labels.astype(str)

    umap_df['dataset'] = ds_labels

    cluster_labels = cluster.KMeans(n_clusters=n_clusters).fit_predict(latent)

    umap_df['cluster'] = cluster_labels

    if plot:
        df = umap_df.copy()
        df['cluster'] = df['cluster'].astype(str)
        sns.scatterplot(data=df, x='x', y='y', hue='cluster')
    return umap_df


def compute_ds_coloc(model: Union[CRL, BioMedCLIP, CVAE], device: str='cpu',
                     origin: Literal['train', 'val', 'test']='train') -> Dict[int, pd.DataFrame]:
    latent = get_latent(model=model, device=device, origin=origin)
    
    # lcos = torch_cosine(torch.tensor(latent))
    # Make DS specific dict
    out_dict = {}        
    data = get_mzimage_dataset(model, origin=origin)
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

def same_ion_similarity(model: CRL, device: str='cpu',
                        origin: Literal['train', 'val', 'test']='train'):

    latent = get_latent(model, origin=origin, device=device)
    ion_labels = get_ion_labels(model, origin=origin) 
    
    # Same ion similarities
    same_similarities = []
    same_ion_counter = 0
    for i in set(ion_labels):
        if len(latent[ion_labels==i]) > 1:
            # Compute cosine of latent reps of all same ions
            a = pairwise_kernels(latent[ion_labels==i], metric='cosine')
            mask = ~np.diag(np.ones(a.shape[0])).astype(bool)
            same_similarities.append(np.mean(a[mask]))
            same_ion_counter += 1
            
    print(f'{same_ion_counter} ions observed in multiple images')
    
    # Compute distance between all Sample N random similarities that are not the same ion
    different_similarities = []
    for i in range(5000):
        samples = np.random.randint(0, high=latent.shape[0], size=2)
        if ion_labels[samples[0]] != ion_labels[samples[1]]:
            a = pairwise_kernels(latent[samples], metric='cosine')
            different_similarities.append(a[1,0])

    # Plot distances as violins
    final_df = pd.concat([pd.DataFrame({'type': 'Same ion', 'similarity': same_similarities}), 
                          pd.DataFrame({'type': 'Different ion', 'similarity': different_similarities})])
    return final_df

