from typing import Dict
import pandas as pd
import numpy as np
import umap

import torch

def randomize_dfs(df_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    
    out = {k: v.copy() for k, v in df_dict.items()}
    for ds, coloc_df in out.items():
        if coloc_df.shape[0] > 0:
            out[ds] = randomize_df(coloc_df)

    return out

def randomize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for r in out.index:
        for c in out.columns:
            if r != c:
                if not pd.isna(out.loc[r, c]):
                    out.loc[r, c] = np.random.uniform(0, 1)
    return out

def compute_umap(latent: np.ndarray) -> pd.DataFrame:
    
    reducer = umap.UMAP()
    data = pd.DataFrame(reducer.fit_transform(latent))
    data = data.rename(columns={0: 'x', 1: 'y'})
    return data

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

def most_colocalized(clc: Dict[int, pd.DataFrame], ds: int) -> np.ndarray:
    gt_coloc = np.array(clc[ds]).copy()
    np.fill_diagonal(gt_coloc, 0)
    max_coloc = np.argmax(gt_coloc, axis=0)
    max_coloc_id = np.array(clc[ds].index[max_coloc])

    return max_coloc_id

def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    return df_copy


def coloc_knn(coloc_matrix: np.ndarray, k: int=30):

    out_matrix = coloc_matrix.copy()

    thresholds = np.sort(out_matrix, axis=1)[:, -k]
    mask = out_matrix < thresholds[:, np.newaxis]

    out_matrix[mask] = 0

    return out_matrix


