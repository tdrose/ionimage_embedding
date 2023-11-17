# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as functional
from ionimage_embedding.models import CLR
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.clr_data import CLRdata
from ionimage_embedding.models.coloc.utils import torch_cosine

# Comment out if not running in interactive mode
%load_ext autoreload
%autoreload 2


# %%

# Check if we are connected to the GPU server
torch.cuda.is_available()


# %%
class LoadingData:
    pass

ds_list = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s',
    '2022-12-07_01h02m53s',
    '2022-12-07_01h01m06s',
    '2022-11-28_22h24m25s',
    '2022-11-28_22h23m30s'
                  ]

clrdat = CLRdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing')


# %%
colocs = ColocModel(clrdat)


# %%
model = CLR(clrdat,
            num_cluster=8,
            initial_upper=93,
            initial_lower=37,
            upper_iteration=.8,
            lower_iteration=.8,
            dataset_specific_percentiles=True,
            knn=True, 
            lr=0.0001,
            pretraining_epochs=10,
            training_epochs=15,
            cae_encoder_dim=2,
            lightning_device='gpu',
            random_seed=1225
            )

device='cuda'
model.train(logger=False)

# %% CRL inference
embds = model.inference_embeddings(new_data=model.data.test_dataset.images, normalize_images=False, device=device)


# %%
def ds_embedding_coloc(colocs: torch.Tensor, ds_labels: torch.Tensor, ion_labels: torch.Tensor) -> dict[int, pd.DataFrame]:
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

embds_ds = ds_embedding_coloc(colocs=torch_cosine(torch.tensor(embds)),
                              ds_labels=model.data.test_dataset.dataset_labels,
                              ion_labels=model.data.test_dataset.ion_labels)



# %%

def sensitivity(x):
    return x['tp'] / (x['tp'] + x['fn'])

def specificity(x):
    return x['tn'] / (x['tn'] + x['fp'])

def accuracy(x):
    return (x['tp'] + x['tn'])/(x['fn']+x['tn']+x['fp']+x['tp'])

def f1score(x):
    return (x['tp']*2)/(x['tp']*2 + x['fp'] + x['fn'])

def precision(x):
    return x['tp'] / (x['tp'] + x['fp'])

def evaluation(evaluation_dict):
     
    lmodel = []
    laccuracy = []
    lf1score = []
    lprecision = []
    lrecall = []
     
    # Evaluate upper
    

    for mod, ds in evaluation_dict['predictions'].items():
        for dsid, eval in ds.items():
            lmodel.append(mod)
            gt = evaluation_dict['ground_truth'][dsid]
            tp = sum([1 for x in eval['upper'] if x in gt['upper']])
            fp = sum([1 for x in eval['upper'] if x not in gt['upper']])
            

            tn = sum([1 for x in eval['lower'] if x in gt['lower']])
            fn = sum([1 for x in eval['lower'] if x not in gt['lower']])
            scores = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            laccuracy.append(accuracy(scores))
            lf1score.append(f1score(scores))
            lprecision.append(precision(scores))
            lrecall.append(sensitivity(scores))

    return pd.DataFrame({'model': lmodel, 
                         'accuracy': laccuracy, 'f1score': lf1score, 
                         'precision': lprecision, 'recall': lrecall,
                         'lq': evaluation_dict['lq'], 'uq': evaluation_dict['uq']})

# %%
coloctesteval = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                     test_dsl=colocs.data.test_dataset.dataset_labels,
                                     test_colocs=colocs.test_mean_coloc, 
                                     upper_quantile=0.9, lower_quantile=0.1)

colocgt = colocs.quantile_gt(test_il = colocs.data.full_dataset.ion_labels, 
                             test_dsl=colocs.data.full_dataset.dataset_labels,
                             full_colocs=colocs.full_coloc, 
                             upper_quantile=0.9, lower_quantile=0.1)

crltesteval = colocs.quantile_gt(full_colocs=embds_ds,
                                 test_dsl=model.data.test_dataset.dataset_labels,
                                 test_il=model.data.test_dataset.ion_labels,
                                 upper_quantile=0.9, lower_quantile=0.1)

evaluation_dict = {'ground_truth': colocgt, 'predictions': {'crl': crltesteval, 'colocmean': coloctesteval}, 'uq': .9, 'lq': .1}
df = evaluation(evaluation_dict)
sns.scatterplot(data=df, x='precision', y='recall', hue='model')


# %%
df_list = []
for quant in np.linspace(0.05, 0.45, 21):

    coloctesteval = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                     test_dsl=colocs.data.test_dataset.dataset_labels,
                                     test_colocs=colocs.test_mean_coloc, 
                                     upper_quantile=1.-quant, lower_quantile=quant)
    
    colocmed = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                     test_dsl=colocs.data.test_dataset.dataset_labels,
                                     test_colocs=colocs.test_median_coloc, 
                                     upper_quantile=1.-quant, lower_quantile=quant)

    colocgt = colocs.quantile_gt(test_il = colocs.data.test_dataset.ion_labels, 
                                 test_dsl=colocs.data.test_dataset.dataset_labels,
                                 full_colocs=colocs.full_coloc, 
                                 upper_quantile=1.-quant, lower_quantile=quant)
    
    crltesteval = colocs.quantile_gt(full_colocs=embds_ds,
                                     test_dsl=model.data.test_dataset.dataset_labels,
                                     test_il=model.data.test_dataset.ion_labels,
                                     upper_quantile=1.-quant, lower_quantile=quant)

    evaluation_dict = {'ground_truth': colocgt, 'predictions': {'crl': crltesteval, 'colocmean': coloctesteval, 
                                                                'colocmedian': colocmed}, 
                       'uq': 1-quant, 'lq': quant}
    
    df_list.append(evaluation(evaluation_dict))
    

# %%
df = pd.concat(df_list).groupby(['model', 'lq', 'uq']).agg('mean').reset_index()

sns.scatterplot(data=df, x='precision', y='recall', hue='model', size='uq')
plt.show()
sns.scatterplot(data=df, x='accuracy', y='f1score', hue='model', size='uq')

# %%
