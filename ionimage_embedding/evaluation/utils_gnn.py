from typing import Literal
import numpy as np

import torch

from ..dataloader.ColocNet_data import ColocNetData_discrete
from ..coloc.coloc import ColocModel


def mean_coloc_test(data: ColocNetData_discrete, agg: Literal['mean', 'median', 'var'] = 'mean'):

    return mean_colocs(data, data._test_set, agg=agg)

def mean_coloc_train(data: ColocNetData_discrete, agg: Literal['mean', 'median', 'var'] = 'mean'):
    
    return mean_colocs(data, data._train_set, agg=agg)


def coloc_ion_labels(data: ColocNetData_discrete, prediction_set: np.ndarray):
    test_ions = []
    for dsid in prediction_set:
        test_ions.extend(data.dataset.coloc_dict[dsid].columns)
    test_ions = torch.tensor(list(set(test_ions)))
    return test_ions
    

def mean_colocs(data: ColocNetData_discrete, prediction_set: np.ndarray, 
                agg: Literal['mean', 'median', 'var'] = 'mean'):
    
    train_data = {k: v for k, v in data.dataset.coloc_dict.items() if k in data.get_train_dsids()}

    test_ions = coloc_ion_labels(data, prediction_set)
    # Compute mean coloc across training datasets
    mean_colocs = ColocModel._inference(ds_colocs=train_data, 
                                        ion_labels=test_ions, agg=agg)
    
    return mean_colocs
