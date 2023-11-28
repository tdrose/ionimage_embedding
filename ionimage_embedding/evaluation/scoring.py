import torch
import pandas as pd
import numpy as np

from .utils import precision, sensitivity, accuracy, f1score

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
