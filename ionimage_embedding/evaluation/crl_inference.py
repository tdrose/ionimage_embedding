import torch
import numpy as np
import pandas as pd
from typing import Tuple

from ..models.coloc.utils import torch_cosine

def crl_latent_coloc(model, dat, device='cpu'):
    
    embds = model.inference_embeddings(new_data=dat.images, normalize_images=False, 
                                                   device=device, use_embed_layer=False)
    print('updated')
    cos = torch_cosine(torch.tensor(embds))

    return pd.DataFrame(cos.cpu().detach().numpy(), 
                        columns=dat.ion_labels.cpu().detach().numpy(),
                        index=dat.ion_labels.cpu().cpu().detach().numpy()
                        )


def crl_fulllatent_coloc(train_latent_coloc, ion_labels: torch.Tensor, 
                         agg: str='mean') -> Tuple[pd.DataFrame, float]:
    # Create torch matrix to fill
    numpy_ion_labels = ion_labels.cpu().detach().numpy()
    numpy_ion_labels = np.unique(numpy_ion_labels)

    out = np.zeros(numpy_ion_labels.shape[0]*\
                   numpy_ion_labels.shape[0]).reshape((numpy_ion_labels.shape[0], -1))

    agg_f = np.mean
    if agg == 'median':
        agg_f = np.median
    elif agg == 'mean':
        pass
    else:
        raise ValueError('Aggregation function not available')
    
    # Sorting the ion labels
    sorted_ion_labels = np.sort(numpy_ion_labels)
    counter = 1e-9
    not_possible = 0.
    for i1 in range(len(sorted_ion_labels)):
        
        for i2 in range(i1, len(sorted_ion_labels)):
            if i1 == i2:
                out[i1, i2] = np.nan
            elif sorted_ion_labels[i1] == sorted_ion_labels[i2]:
                out[i1, i2] = np.nan
                out[i2, i1] = np.nan
            else:
                # Loop over all datasets to check if coloc has been observed before
                aggs = []
                ion1 = int(sorted_ion_labels[i1])
                ion2 = int(sorted_ion_labels[i2])
                counter += 1

                if ion1 in train_latent_coloc.columns and ion2 in train_latent_coloc.columns:
                    out[i1, i2] = agg_f(train_latent_coloc.loc[ion1, ion2])
                    out[i2, i1] = agg_f(train_latent_coloc.loc[ion1, ion2])
                else:
                    not_possible += 1.
                    out[i1, i2] = np.nan
                    out[i2, i1] = np.nan

    not_inferred_fraction = not_possible / counter

    return pd.DataFrame(out, 
                        columns=sorted_ion_labels, 
                        index=sorted_ion_labels), not_inferred_fraction



def crl_ds_coloc(model, dat, device='cpu') -> dict[int, pd.DataFrame]:
        out_dict = {}        

        # loop over each dataset
        for dsl in torch.unique(dat.dataset_labels):
            dsid = int(dsl)
            mask = dat.dataset_labels==dsid
            if sum(mask) > 1:
                # Convert data to array (in correct order)
                imgs = torch.tensor(dat.images)[mask]
                ions = dat.ion_labels[mask]
                ion_sorting_mask = torch.argsort(ions)
                imgs = imgs[ion_sorting_mask]
                # Compute coloc matrix (in correct order)
                embds = model.inference_embeddings(new_data=imgs, normalize_images=False, 
                                                   device=device, use_embed_layer=True)
                cos = torch_cosine(torch.tensor(embds))
                out_dict[dsid] = pd.DataFrame(cos.cpu().detach().numpy(), 
                                              columns=ions[ion_sorting_mask].cpu().detach().numpy(),
                                              index=ions[ion_sorting_mask].cpu().detach().numpy()
                                              )
            else:
                out_dict[dsid] = pd.DataFrame()

        return out_dict

def crl_latent_inference(train_latent_coloc, ion_labels: torch.Tensor, 
                         agg: str='mean') -> Tuple[pd.DataFrame, float]:

        # Create torch matrix to fill
        numpy_ion_labels = ion_labels.cpu().detach().numpy()
        numpy_ion_labels = np.unique(numpy_ion_labels)

        out = np.zeros(numpy_ion_labels.shape[0]*\
                       numpy_ion_labels.shape[0]).reshape((numpy_ion_labels.shape[0], -1))

        agg_f = np.mean
        if agg == 'median':
            agg_f = np.median
        elif agg == 'mean':
            pass
        else:
            raise ValueError('Aggregation function not available')
        
        # Sorting the ion labels
        sorted_ion_labels = np.sort(numpy_ion_labels)
        counter = 1e-9
        not_possible = 0.
        for i1 in range(len(sorted_ion_labels)):
            
            for i2 in range(i1, len(sorted_ion_labels)):
                if i1 == i2:
                    out[i1, i2] = np.nan
                elif sorted_ion_labels[i1] == sorted_ion_labels[i2]:
                    out[i1, i2] = np.nan
                    out[i2, i1] = np.nan
                else:
                    # Loop over all datasets to check if coloc has been observed before
                    aggs = []
                    ion1 = int(sorted_ion_labels[i1])
                    ion2 = int(sorted_ion_labels[i2])
                    counter += 1.
                    checker = True
                    for ds in train_latent_coloc.keys():
                        # Check if ion pair was co-detected in any of the training data
                        if ion1 in train_latent_coloc[ds].columns and \
                            ion2 in train_latent_coloc[ds].columns:
                            aggs.append(train_latent_coloc[ds].loc[ion1, ion2])
                            checker = False
                    if checker:
                        not_possible += 1.
                    
                    if len(aggs) > 0:
                        out[i1, i2] = agg_f(aggs)
                        out[i2, i1] = agg_f(aggs)
                    else:
                        out[i1, i2] = np.nan
                        out[i2, i1] = np.nan

        not_inferred_fraction = not_possible / counter

        return pd.DataFrame(out, 
                            columns=sorted_ion_labels, 
                            index=sorted_ion_labels), not_inferred_fraction

