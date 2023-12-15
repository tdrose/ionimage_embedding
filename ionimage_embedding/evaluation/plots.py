import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import seaborn as sns

from .utils import get_latent, get_ds_labels, compute_umap, get_ion_images
from ..models.crl.crl import CRL



def umap_allorigin(model: CRL, device='cpu', 
                   fig: Optional[Figure]=None) -> Tuple[Figure, Axes, Axes]:

    train_latent = get_latent(model=model, device=device, origin='train')
    val_latent = get_latent(model=model, device=device, origin='val')
    test_latent = get_latent(model=model, device=device, origin='test')

    all_data = np.concatenate((train_latent, val_latent, test_latent))
    umap_df = compute_umap(all_data)

    all_ds_labels = np.concatenate((get_ds_labels(model=model, origin='train'),
                                    get_ds_labels(model=model, origin='val'),
                                    get_ds_labels(model=model, origin='test')))
    all_ds_labels = all_ds_labels.astype(str)

    umap_df['dataset'] = all_ds_labels
    umap_df['origin'] = np.concatenate((np.repeat('train', train_latent.shape[0]),
                                        np.repeat('val', val_latent.shape[0]),
                                        np.repeat('test', test_latent.shape[0])))
    
    if fig is None:
        fig, (ax0, ax1) = plt.subplots(ncols=2)
    else:
        (ax0, ax1) = fig.subplots(ncols=2)  # type: ignore
    
    sns.scatterplot(data=umap_df, x='x', y='y', ax=ax0, hue='origin')
    sns.scatterplot(data=umap_df, x='x', y='y', ax=ax1, hue='dataset')

    return fig, ax0, ax1

def umap_latent(model: CRL, device='cpu',
                origin: Literal['train', 'val', 'test']='train',
                fig: Optional[Figure]=None) -> Tuple[Figure, Axes]:

    latent = get_latent(model=model, device=device, origin=origin)
    umap_df = compute_umap(latent)

    ds_labels = get_ds_labels(model=model, origin=origin)
    ds_labels = ds_labels.astype(str)

    umap_df['dataset'] = ds_labels
    
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.subplots()
    
    sns.scatterplot(data=umap_df, x='x', y='y', ax=ax, hue='dataset')

    return fig, ax



def plot_image_clusters(model: CRL, cluster_df: pd.DataFrame, 
                        origin: Literal['train', 'val', 'test']='train'):
    
    images = get_ion_images(model=model, origin=origin)
    ds_labels = get_ds_labels(model=model, origin=origin)
    
    # row and col mapping:
    ds_mapping = {value: count for count, value in enumerate(set(ds_labels))}
    cluster_mapping = {value: count for count, value in enumerate(set(cluster_df['cluster']))}

    # Create Figure:
    fig, axs = plt.subplots(nrows=max([x for x in ds_mapping.values()])+1,
                            ncols=max([x for x in cluster_mapping.values()])+1)

    for ds in set(ds_labels):

        ds_mask = ds_labels == ds
        ds_images = images[ds_mask]
        ds_clusters = np.array(cluster_df['cluster'])[ds_mask]

        for cluster in set(cluster_df['cluster']):

            mean_image = np.mean(ds_images[ds_clusters==cluster], axis=0)
            ax = axs[ds_mapping[ds], cluster_mapping[cluster]]
            ax.imshow(mean_image)
            ax.set_title(f'Dataset {ds}, Cluster {cluster}', fontdict={'fontsize': 5})

    axs_flat = axs.flatten()
    for ax in axs_flat:
        ax.axis('off')

    plt.show()
