from typing import Dict, Tuple
import pandas as pd
import numpy as np

from ._utils import most_colocalized

def sensitivity(x: Dict) -> float:
    return x['tp'] / (x['tp'] + x['fn'])

def specificity(x: Dict) -> float:
    return x['tn'] / (x['tn'] + x['fp'])

def accuracy(x: Dict) -> float:
    return (x['tp'] + x['tn'])/(x['fn']+x['tn']+x['fp']+x['tp'])

def f1score(x: Dict) -> float:
    return (x['tp']*2)/(x['tp']*2 + x['fp'] + x['fn'])

def precision(x: Dict) -> float:
    return x['tp'] / (x['tp'] + x['fp'])


def coloc_mse(latent: pd.DataFrame, agg_coloc_pred: pd.DataFrame, 
              true_df_dict: Dict[int, pd.DataFrame]) -> Tuple[float, float, float]:
    # Loop over all elements of agg_coloc_pred using their column and index names
    avail_total = 0
    trans_total = 0
    avail_error = 0
    trans_error = 0

    for ds, coloc_df in true_df_dict.items():
        if coloc_df.shape[0] > 0:
            
            for r in coloc_df.index:
                for c in coloc_df.columns:
                    if r != c:
                        if not pd.isna(latent.loc[r, c]):
                            # Transitivity
                            if pd.isna(agg_coloc_pred.loc[r, c]):
                                trans_error += (coloc_df.loc[r, c] - 
                                                latent.loc[r, c])**2 # type: ignore
                                trans_total += 1
                            # Availability
                            else:
                                avail_error += (coloc_df.loc[r, c] - 
                                                latent.loc[r, c])**2 # type: ignore
                                avail_total += 1

    if avail_total == 0:
        avail = np.nan
    else:
        avail = avail_error / avail_total

    if trans_total == 0:
        trans = np.nan
    else:
        trans = trans_error / trans_total

    if np.isnan(avail) or np.isnan(trans):
        fraction = np.nan
    else:
        fraction = avail_total/(trans_total+avail_total)
    
    return avail, trans, fraction

def coloc_top_acc(latent: pd.DataFrame, 
                  true_df_dict: Dict[int, pd.DataFrame], 
                  aggcoloc: pd.DataFrame,
                  top: int=5) -> Tuple[float, float, float]:

    avail_corr = 0
    avail_total = 0
    trans_corr = 0
    trans_total = 0

    clc = true_df_dict

    pred_df = latent
    for ds, coloc_df in clc.items():
        if clc[ds].shape[0] > 0:
            
            # Get most colocalized image per image per dataset
            max_coloc_id = most_colocalized(clc=clc, ds=ds)

            # create dataset specific coloc df from mean colocs
            curr_cl = np.array(pred_df.loc[clc[ds].index, clc[ds].index]).copy() # type: ignore
            
            # Set diagonal to zero (no self coloc)
            np.fill_diagonal(curr_cl, 0)
            
            # Set NA values to zero
            curr_cl[np.isnan(curr_cl)] = 0

            # Loop over each molecule
            for i in range(len(curr_cl)):
                
                # Only evaluate if molecule has been observed in training data
                if not all(curr_cl[i] == 0):

                    # Descending sorted most colocalized
                    mask = np.argsort(curr_cl[i])[::-1]

                    # We are using the coloc_df index for the curr_cl array
                    coloc_order = coloc_df.index[mask]

                    if np.isnan(aggcoloc.loc[clc[ds].index[i], 
                                             max_coloc_id[i]]) and \
                        not np.isnan(pred_df.loc[clc[ds].index[i], max_coloc_id[i]]):
                        
                        if max_coloc_id[i] in coloc_order[:top]:
                            trans_corr += 1

                        trans_total += 1
                    elif not np.isnan(aggcoloc.loc[clc[ds].index[i], max_coloc_id[i]]):
                        if max_coloc_id[i] in coloc_order[:top]:
                            avail_corr += 1

                        avail_total += 1

    if avail_total == 0:
        avail = np.nan
    else:
        avail = avail_corr / avail_total

    if trans_total == 0:
        trans = np.nan
    else:
        trans = trans_corr / trans_total

    if np.isnan(avail) or np.isnan(trans):
        fraction = np.nan
    else:
        fraction = avail_total/(trans_total+avail_total)
    
    return avail, trans, fraction
