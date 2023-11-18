import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f

from ...dataloader.clr_data import CLRdata
from .utils import torch_cosine, quantile_sets

class ColocModel:
    def __init__(self, data: CLRdata, device='cpu') -> None:
        self.data = data
        self.device = device

        # Compute full data cossine
        self.full_coloc = self.compute_ds_coloc(self.data.full_dataset)

        # Compute train data cosine
        self.train_coloc = self.compute_ds_coloc(self.data.full_dataset)

        # Compute test data mean/median cosine
        self.test_mean_coloc = self.inference(self.data.test_dataset.ion_labels, agg='mean')
        self.test_median_coloc = self.inference(self.data.test_dataset.ion_labels, agg='median')
    
    @staticmethod
    def quantile_eval(test_il: torch.Tensor, test_dsl: torch.Tensor, 
                      test_colocs: pd.DataFrame, 
                      upper_quantile: float=0.9, lower_quantile: float=0.1) -> dict[int, dict]:
        out_dict = {}

        for dsl in torch.unique(test_dsl):
            dsid = int(dsl)
            mask = test_dsl==dsid
            if sum(mask) > 1:
                masked_ill = test_il[mask].cpu().detach().numpy()
                numpy_ion_labels = np.unique(masked_ill)
                sorted_ion_labels = np.sort(numpy_ion_labels)
                cdf = test_colocs.loc[sorted_ion_labels, sorted_ion_labels].copy()

                out_dict[dsid] = quantile_sets(cdf, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

        return out_dict
    
    @staticmethod
    def quantile_gt(test_il: torch.Tensor, test_dsl: torch.Tensor,
                    full_colocs: dict[int, pd.DataFrame], 
                    upper_quantile: float=0.9, lower_quantile: float=0.1) -> dict[int, dict]:
        
        out_dict = {}
        for dsid in full_colocs.keys():
                # Subsetting ground truth
                mask = test_dsl==dsid
                masked_ill = test_il[mask].cpu().detach().numpy()
                numpy_ion_labels = np.unique(masked_ill)
                sorted_ion_labels = np.sort(numpy_ion_labels)
                cdf = full_colocs[dsid].loc[sorted_ion_labels, sorted_ion_labels].copy()

                cdf_numpy = np.array(cdf)
                np.fill_diagonal(cdf_numpy, np.nan)
                cdf = pd.DataFrame(cdf_numpy, columns=cdf.columns, index=cdf.index)

                out_dict[dsid] = quantile_sets(cdf, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

        return out_dict
    
    def compute_ds_coloc(self, dat) -> dict[int, pd.DataFrame]:
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
                cos = torch_cosine(imgs)
                out_dict[dsid] = pd.DataFrame(cos.cpu().detach().numpy(), 
                                              columns=ions[ion_sorting_mask].cpu().detach().numpy(),
                                              index=ions[ion_sorting_mask].cpu().detach().numpy()
                                              )
            else:
                out_dict[dsid] = pd.DataFrame()

        return out_dict
    
    def inference(self, ion_labels: torch.Tensor, agg: str='mean') -> pd.DataFrame:

        # Create torch matrix to fill
        numpy_ion_labels = ion_labels.cpu().detach().numpy()
        numpy_ion_labels = np.unique(numpy_ion_labels)

        out = np.zeros(numpy_ion_labels.shape[0]*numpy_ion_labels.shape[0]).reshape((numpy_ion_labels.shape[0], -1))

        agg_f = np.mean
        if agg == 'median':
            agg_f = np.median
        elif agg == 'mean':
            pass
        else:
            raise ValueError('Aggregation function not available')
        
        # Sorting the ion labels
        sorted_ion_labels = np.sort(numpy_ion_labels)

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
                    for ds in self.train_coloc.keys():
                        if ion1 in self.train_coloc[ds].columns and ion2 in self.train_coloc[ds].columns:
                            aggs.append(self.train_coloc[ds].loc[ion1, ion2])
                    
                    if len(aggs) > 0:
                        out[i1, i2] = agg_f(aggs)
                        out[i2, i1] = agg_f(aggs)
                    else:
                        out[i1, i2] = np.nan
                        out[i2, i1] = np.nan

        return pd.DataFrame(out, 
                            columns=sorted_ion_labels,
                            index=sorted_ion_labels
                            )