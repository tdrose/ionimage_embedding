import torch
import pandas as pd
import numpy as np
from typing import Dict, Literal
from sklearn.metrics import silhouette_score

from .utils import precision, sensitivity, accuracy, f1score
from ..dataloader.crl_dataloader import mzImageDataset
from ..models.crl.crl import CRL
from ..models.coloc.coloc import ColocModel
from ..models.coloc.utils import torch_cosine

def ds_coloc_convert(colocs: torch.Tensor, ds_labels: torch.Tensor, ion_labels: torch.Tensor) -> dict[int, pd.DataFrame]:
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
                              dataset: Literal['train', 'val', 'test']='train', device: str='cpu'):
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


def get_mzimage_dataset(model: CRL, dataset: Literal['train', 'val', 'test']='train') -> mzImageDataset:
    if dataset == 'train':
        return model.data.train_dataset
    elif dataset == 'test':
        return model.data.test_dataset
    elif dataset == 'val':
        return model.data.val_dataset
    else:
        raise ValueError("`dataset` must be one of: ['train', 'val', 'test']")


def compute_ds_coloc(model: CRL, device: str='cpu',
                     dataset: Literal['train', 'val', 'test']='train') -> Dict[int, pd.DataFrame]:
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
    data = get_mzimage_dataset(model, dataset=dataset)
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


def closest_accuracy_random(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                            dataset: Literal['train', 'val', 'test']='train') -> float:

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


def closest_accuracy_latent(ds_coloc_dict: Dict[int, pd.DataFrame], colocs: ColocModel, top: int=5, 
                            dataset: Literal['train', 'val', 'test']='train') -> float:

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


def closest_accuracy_aggcoloc(colocs: ColocModel, top: int=5,
                              agg: Literal['mean', 'median']='mean') -> float:

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
