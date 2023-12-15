from typing import Literal, Tuple
import numpy as np
import pandas as pd
import umap
import seaborn as sns
import sklearn.cluster as cluster

from ..models.crl.crl import CRL
from ..dataloader.crl_dataloader import mzImageDataset

def sensitivity(x: dict) -> float:
    return x['tp'] / (x['tp'] + x['fn'])

def specificity(x: dict) -> float:
    return x['tn'] / (x['tn'] + x['fp'])

def accuracy(x: dict) -> float:
    return (x['tp'] + x['tn'])/(x['fn']+x['tn']+x['fp']+x['tp'])

def f1score(x: dict) -> float:
    return (x['tp']*2)/(x['tp']*2 + x['fp'] + x['fn'])

def precision(x: dict) -> float:
    return x['tp'] / (x['tp'] + x['fp'])

def get_mzimage_dataset(model: CRL, 
                        origin: Literal['train', 'val', 'test']='train') -> mzImageDataset:
    if origin == 'train':
        return model.data.train_dataset
    elif origin == 'test':
        return model.data.test_dataset
    elif origin == 'val':
        return model.data.val_dataset
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")

def get_latent(model: CRL, device: str='cpu',
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

def get_ds_labels(model: CRL,
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

def get_ion_labels(model: CRL,
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

def get_ion_images(model: CRL,
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

def latent_centroids(model, 
                     origin: Literal['train', 'val', 'test']='train') -> Tuple[np.ndarray, 
                                                                               np.ndarray]:

    latent = get_latent(model=model, origin=origin)
    ion_labels = get_ion_labels(model=model, origin=origin)

    ion_centroids = []
    centroid_labels = []
    for i in set(ion_labels):
        if len(latent[ion_labels==i]) > 1:
            a = latent[ion_labels==i]
            ion_centroids.append(a.sum(axis=0)/a.shape[0])
            centroid_labels.append(i)
        else:
            ion_centroids.append(latent[ion_labels==i][0])
            centroid_labels.append(i)
            
    return np.stack(ion_centroids), np.array(centroid_labels)


def compute_umap(latent: np.ndarray) -> pd.DataFrame:
    
    reducer = umap.UMAP()
    data = pd.DataFrame(reducer.fit_transform(latent))
    data = data.rename(columns={0: 'x', 1: 'y'})
    return data


def cluster_latent(model, n_clusters: int=10, plot: bool=False, device='cpu',
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
