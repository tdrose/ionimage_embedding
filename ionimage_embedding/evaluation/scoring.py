import torch
import pandas as pd
import numpy as np
from typing import Dict, Literal, Union
from sklearn.metrics import silhouette_score, pairwise_kernels
from anndata import AnnData
import scanpy as sc
from scipy import sparse

from .utils import (
    precision, 
    sensitivity, 
    accuracy, 
    f1score,
    get_ds_labels,
    get_ion_labels,
    get_latent,
    get_mzimage_dataset)
from ..dataloader.crl_dataloader import mzImageDataset
from ..models.crl.crl import CRL
from ..models.coloc.coloc import ColocModel
from ..models.coloc.utils import torch_cosine
from ..models.biomedclip import BioMedCLIP

def ds_coloc_convert(colocs: torch.Tensor, 
                     ds_labels: torch.Tensor, ion_labels: torch.Tensor) -> dict[int, pd.DataFrame]:
        out_dict = {}        

        # loop over each dataset
        for dsl in torch.unique(ds_labels):
            dsid = int(dsl)
            mask = ds_labels==dsid
            if sum(mask) > 1:
                # df for easier indexing
                ions = ion_labels[mask]
                ds_colocs = colocs[mask, :][:, mask].cpu().detach().numpy()

                np.fill_diagonal(ds_colocs, np.nan)

                df = pd.DataFrame(ds_colocs, 
                                  columns=ions.cpu().detach().numpy(),
                                  index=ions.cpu().detach().numpy()
                                  )

                out_dict[dsid] = df
            else:
                out_dict[dsid] = pd.DataFrame()

        return out_dict


def evaluation_quantile_overlap(evaluation_dict):
     
    lmodel = []
    laccuracy = []
    lf1score = []
    lprecision = []
    lrecall = []
     
    # Evaluate upper
    

    for mod, ds in evaluation_dict['predictions'].items():
        for dsid, eval in ds.items():
            
            gt = evaluation_dict['ground_truth'][dsid]
            tp = sum([1 for x in eval['upper'] if x in gt['upper']])
            fp = sum([1 for x in eval['upper'] if x not in gt['upper']])
            

            tn = sum([1 for x in eval['lower'] if x in gt['lower']])
            fn = sum([1 for x in eval['lower'] if x not in gt['lower']])
            scores = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            lmodel.append(mod)
            laccuracy.append(accuracy(scores))
            lf1score.append(f1score(scores))
            lprecision.append(precision(scores))
            lrecall.append(sensitivity(scores))

    return pd.DataFrame({'model': lmodel, 
                         'accuracy': laccuracy, 'f1score': lf1score, 
                         'precision': lprecision, 'recall': lrecall,
                         'lq': evaluation_dict['lq'], 'uq': evaluation_dict['uq']})


def latent_dataset_silhouette(model: CRL, metric: str='cosine', 
                              origin: Literal['train', 'val', 'test']='train', device: str='cpu'):
    if origin == 'train':
        latent = model.inference_embeddings_train(device=device)
        ds_labels = model.data.train_dataset.dataset_labels.detach().cpu().numpy()
    elif origin == 'test':
        latent = model.inference_embeddings_test(device=device)
        ds_labels = model.data.test_dataset.dataset_labels.detach().cpu().numpy()
    elif origin == 'val':
        latent = model.inference_embeddings_val(device=device)
        ds_labels = model.data.val_dataset.dataset_labels.detach().cpu().numpy()
    else:
        raise ValueError("`origin` must be one of: ['train', 'val', 'test']")
    
    return silhouette_score(X=latent, labels=ds_labels, metric=metric) 


def compute_ds_coloc(model: Union[CRL, BioMedCLIP], device: str='cpu',
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


def get_colocs(colocs: ColocModel, origin: Literal['train', 'val', 'test']='train'):
    if origin == 'train':
        return colocs.train_coloc
    elif origin == 'test':
        return colocs.test_coloc
    elif origin == 'val':
        return colocs.val_coloc
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")

def most_colocalized(clc: Dict[int, pd.DataFrame], ds: int) -> np.ndarray:
    gt_coloc = np.array(clc[ds]).copy()
    np.fill_diagonal(gt_coloc, 0)
    max_coloc = np.argmax(gt_coloc, axis=0)
    max_coloc_id = np.array(clc[ds].index[max_coloc])

    return max_coloc_id


def closest_accuracy_random(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                            origin: Literal['train', 'val', 'test']='train') -> float:

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, origin=origin)
    for ds, coloc_df in ds_coloc_dict.items():
        if clc[ds].shape[0] > 0:
            # Get most colocalized image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

            # convert coloc_df to numpy array
            latent_coloc = np.array(coloc_df).copy()
            np.fill_diagonal(latent_coloc, 0)

            for i in range(len(latent_coloc)):

                # Descending sorted most colocalized
                size = np.min([coloc_df.shape[0], top])
                coloc_order = np.random.choice(coloc_df.index, size, replace=False)

                if max_coloc_id[i] in coloc_order:
                    correct_predictions += 1

                total_predictions += 1

    return correct_predictions / total_predictions


def closest_accuracy_latent(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                            origin: Literal['train', 'val', 'test']='train') -> float:

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, origin=origin)
    for ds, coloc_df in ds_coloc_dict.items():
        if clc[ds].shape[0] > 0:
            # Get most colocalized image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

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


def closest_accuracy_aggcoloc(colocs: ColocModel, top: int=5,
                              agg: Literal['mean', 'median']='mean') -> float:

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, origin='test')

    if agg == 'mean':
        pred_df = colocs.test_mean_coloc
    elif agg == 'median':
        pred_df = colocs.test_median_coloc
    else:
        raise ValueError("`agg` must be one of: ['mean', 'median']")

    for ds, coloc_df in clc.items():
        if clc[ds].shape[0] > 0:
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

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

def coloc_knn(coloc_matrix: np.ndarray, k: int=30):

    out_matrix = coloc_matrix.copy()

    thresholds = np.sort(out_matrix, axis=1)[:, -k]
    mask = out_matrix < thresholds[:, np.newaxis]

    out_matrix[mask] = 0

    return out_matrix

def coloc_umap(colocs: ColocModel, k: int=3, n_components: int=10) -> pd.DataFrame:

    labels = torch.unique(colocs.data.train_dataset.ion_labels)
    mean_coloc, fraction = colocs.inference(labels)
    mean_coloc[np.isnan(mean_coloc)] = 0
    # Create anndata object

    coloc_adata = AnnData(X=mean_coloc)

    # Create nearest neighbor matrix (with n nearest neighbors weighted by coloc)
    coloc_adata.uns['neighbors'] = {'params': {'method': 'umap', 
                                            'metric': 'cosine'},
                                    'connectivities_key': 'connectivities'}

    coloc_adata.obsp['connectivities'] = sparse.csr_array(
        coloc_knn(coloc_adata.X, k=k) # type: ignore
    )

    # run scnapy umap function
    sc.tl.umap(coloc_adata, n_components=n_components)

    return pd.DataFrame(coloc_adata.obsm['X_umap'], 
                        index=labels.detach().cpu().numpy())

def umap_inference(umap_df: pd.DataFrame, test_labels: torch.Tensor):
    
    sim = pd.DataFrame(torch_cosine(torch.tensor(np.array(umap_df))).detach().cpu().numpy(), 
                       index=umap_df.index, columns=umap_df.index)
    
    sorted_ion_labels = np.sort(np.unique(test_labels.detach().cpu().numpy()))
    out = np.zeros((len(sorted_ion_labels), len(sorted_ion_labels)))
    
    for i1 in range(len(sorted_ion_labels)):
            for i2 in range(i1, len(sorted_ion_labels)):
                if i1 == i2:
                    out[i1, i2] = np.nan
                elif sorted_ion_labels[i1] == sorted_ion_labels[i2]:
                    out[i1, i2] = np.nan
                    out[i2, i1] = np.nan
                else:
                    if sorted_ion_labels[i1] in sim.index and sorted_ion_labels[i2] in sim.index:
                        out[i1, i2] = sim.loc[sorted_ion_labels[i1], sorted_ion_labels[i2]]
                        out[i2, i1] = sim.loc[sorted_ion_labels[i1], sorted_ion_labels[i2]]
                    else:
                        out[i1, i2] = np.nan
                        out[i2, i1] = np.nan
    return pd.DataFrame(out, index=sorted_ion_labels, columns=sorted_ion_labels)

def closest_accuracy_umapcoloc(umap_inferred: pd.DataFrame, colocs: ColocModel, top: int=5) -> float:

    total_predictions = 0
    correct_predictions = 0
    clc = get_colocs(colocs, origin='test')

    pred_df = umap_inferred
    for ds, coloc_df in clc.items():
        if clc[ds].shape[0] > 0:
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

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
