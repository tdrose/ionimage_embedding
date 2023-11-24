# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns

import torch
import torch.nn.functional as functional

from ionimage_embedding.models import CRL
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.crl_data import CRLdata
from ionimage_embedding.models.coloc.utils import torch_cosine
from ionimage_embedding.evaluation import evaluation_quantile_overlap, ds_coloc_convert
# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
except ImportError:
    # Not in IPython, continue with normal Python code
    pass


# %%

# Check if we are connected to the GPU server
import os
os.system('nvidia-smi')

# %%


# %%
class LoadingData:
    pass

# Kidney datasets
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

# Brain datasets
# ds_list = [
#     '2016-09-21_16h06m55s',
#     '2016-09-21_16h06m56s',
#     '2017-02-17_14h41m43s',
#     '2017-02-17_14h56m37s',
#     '2017-02-24_13h22m14s',
#     '2021-11-04_11h38m42s',
#     '2021-11-04_14h12m55s',
#     '2021-11-11_11h49m37s',
#     '2022-05-30_20h44m19s',
#     '2022-05-31_10h27m17s',
#     '2022-05-31_10h46m34s',
#     '2022-07-19_19h29m24s'
#             ]

clrdat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, fdr=.1)

print(np.isnan(clrdat.full_dataset.images).any())
plt.imshow(clrdat.full_dataset.images[10])
plt.show()


# %%
colocs = ColocModel(clrdat)


# %%
model = CRL(clrdat,
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
            cae=False,
            activation='sigmoid',
            clip_gradients=None
            )

device='cuda'
model.train(logger=False)

# %% CRL inference
embds = model.inference_embeddings(new_data=model.data.test_dataset.images, normalize_images=False, device=device, use_embed_layer=True)


# %%
embds_ds = ds_coloc_convert(colocs=torch_cosine(torch.tensor(embds)),
                            ds_labels=model.data.test_dataset.dataset_labels,
                            ion_labels=model.data.test_dataset.ion_labels)

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
df = evaluation_quantile_overlap(evaluation_dict)
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
    
    df_list.append(evaluation_quantile_overlap(evaluation_dict))
    

# %%
df = pd.concat(df_list).groupby(['model', 'lq', 'uq']).agg('mean').reset_index()

ax = sns.scatterplot(data=df, x='precision', y='recall', hue='model', size='uq')
ax.axline((1, 1), slope=1)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
plt.show()
sns.scatterplot(data=df, x='accuracy', y='f1score', hue='model', size='uq')

# %%
colocs.test_mean_ni
# %%
colocs.test_median_ni
# %%
