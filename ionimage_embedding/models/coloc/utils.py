import torch
import torch.nn.functional as f
import pandas as pd
import numpy as np

def torch_cosine(mat: torch.Tensor, dim=1):
    # Reshaping
    tmp_dat = mat.reshape((mat.shape[0], -1))
    # Normalizing length of each vector to 1
    a = f.normalize(tmp_dat, p=2, dim=dim)
    # computing coloc
    return torch.mm(a, a.transpose(0, 1))


def quantile_sets(cdf: pd.DataFrame, upper_quantile: float=0.9, lower_quantile: float=0.1):
    
    tmp = cdf.melt(var_name='ion2', value_name='val', ignore_index=False).reset_index().rename(columns={'index': 'ion1'})
    tmp = tmp[~np.isnan(tmp['val'])]
    lq = np.quantile(tmp['val'], lower_quantile)
    uq = np.quantile(tmp['val'], upper_quantile)

    upper = tmp[tmp['val'] >= uq]
    upper.loc[:, 'pair'] = upper[['ion1', 'ion2']].apply(lambda x: tuple(sorted(x)), axis=1)
    lower = tmp[tmp['val'] <= lq]
    lower.loc[:, 'pair'] = lower[['ion1', 'ion2']].apply(lambda x: tuple(sorted(x)), axis=1)
    return {'upper': list(set(upper['pair'])), 'lower': list(set(lower['pair']))}