import pandas as pd
from typing import Literal, Union, Tuple
from sklearn.metrics import silhouette_score

from ..dataloader.ColocNet_data import ColocNetData_discrete
from ..coloc.coloc import ColocModel
from ..models import CRL, BioMedCLIP, CVAE
from ._metrics import (
     coloc_mse, 
     coloc_top_acc, 
     accuracy, 
     f1score, 
     precision, 
     sensitivity, 
     coloc_mae, 
     coloc_smape)
from ._utils import randomize_df
from .utils_iid import get_colocs

def coloc_mse_gnn(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                 data: ColocNetData_discrete) -> Tuple[float, float, float]:
    
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items() 
                    if k in data.get_test_dsids()}

    return coloc_mse(latent, agg_coloc_pred, ground_truth)


def coloc_mse_gnn_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                 data: ColocNetData_discrete) -> Tuple[float, float, float]:
    
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items() 
                    if k in data.get_test_dsids()}
    
    return coloc_mse(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_mse_iid(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  colocs: ColocModel) -> Tuple[float, float, float]:
    
    ground_truth = get_colocs(colocs, origin='test')

    return coloc_mse(latent, agg_coloc_pred, ground_truth)


def coloc_mse_iid_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                         colocs: ColocModel) -> Tuple[float, float, float]:

    ground_truth = get_colocs(colocs, origin='test')

    return coloc_mse(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_mae_gnn(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  data: ColocNetData_discrete) -> Tuple[float, float, float]:
     
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items()
                    if k in data.get_test_dsids()}
    
    return coloc_mae(latent, agg_coloc_pred, ground_truth)


def coloc_mae_gnn_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  data: ColocNetData_discrete) -> Tuple[float, float, float]:
     
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items()
                    if k in data.get_test_dsids()}
    
    return coloc_mae(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_mae_iid(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  colocs: ColocModel) -> Tuple[float, float, float]:
    
    ground_truth = get_colocs(colocs, origin='test')

    return coloc_mae(latent, agg_coloc_pred, ground_truth)


def coloc_mae_iid_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                         colocs: ColocModel) -> Tuple[float, float, float]:

    ground_truth = get_colocs(colocs, origin='test')

    return coloc_mae(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_smape_gnn(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  data: ColocNetData_discrete) -> Tuple[float, float, float]:
     
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items()
                    if k in data.get_test_dsids()}
    
    return coloc_smape(latent, agg_coloc_pred, ground_truth)


def coloc_smape_gnn_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  data: ColocNetData_discrete) -> Tuple[float, float, float]:
     
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items()
                    if k in data.get_test_dsids()}
    
    return coloc_smape(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_smape_iid(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                  colocs: ColocModel) -> Tuple[float, float, float]:
    
    ground_truth = get_colocs(colocs, origin='test')

    return coloc_smape(latent, agg_coloc_pred, ground_truth)


def coloc_smape_iid_random(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame,
                         colocs: ColocModel) -> Tuple[float, float, float]:

    ground_truth = get_colocs(colocs, origin='test')

    return coloc_smape(randomize_df(latent), agg_coloc_pred, ground_truth)


def coloc_top_acc_gnn(latent: pd.DataFrame, data: ColocNetData_discrete,
                      agg_coloc_pred: pd.DataFrame, top: int=5) -> Tuple[float, float, float]:
    
    ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items() 
                    if k in data.get_test_dsids()}
    
    return coloc_top_acc(latent, ground_truth, agg_coloc_pred, top)


def coloc_top_acc_gnn_random(latent: pd.DataFrame, data: ColocNetData_discrete,
                             agg_coloc_pred: pd.DataFrame, 
                             top: int=5) -> Tuple[float, float, float]:
        
        ground_truth = {k: v.copy() for k, v in data.dataset.coloc_dict.items() 
                        if k in data.get_test_dsids()}
        
        return coloc_top_acc(randomize_df(latent), ground_truth, agg_coloc_pred, top)


def coloc_top_acc_iid(latent: pd.DataFrame, colocs: ColocModel,
                      agg_coloc_pred: pd.DataFrame, top: int=5) -> Tuple[float, float, float]:
        
        ground_truth = get_colocs(colocs, origin='test')
    
        return coloc_top_acc(latent, ground_truth, agg_coloc_pred, top)

def coloc_top_acc_iid_random(latent: pd.DataFrame, colocs: ColocModel,
                             agg_coloc_pred: pd.DataFrame, 
                             top: int=5) -> Tuple[float, float, float]:
        
        ground_truth = get_colocs(colocs, origin='test')
    
        return coloc_top_acc(randomize_df(latent), ground_truth, agg_coloc_pred, top)



# DEPRECATED
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


def silhouette_iid(model: Union[CRL, BioMedCLIP, CVAE], metric: str='cosine', 
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
