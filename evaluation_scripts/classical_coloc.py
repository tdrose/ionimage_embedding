# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

import torch

from ionimage_embedding.models import CRL, CRL2, CRL3
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.crl_data import CRLdata
from ionimage_embedding.models.coloc.utils import torch_cosine
from ionimage_embedding.evaluation import evaluation_quantile_overlap, ds_coloc_convert
from ionimage_embedding.evaluation.crl_inference import crl_ds_coloc, crl_fulllatent_coloc, crl_latent_coloc, crl_latent_inference
from ionimage_embedding.dataloader.utils import pairwise_same_elements
from ionimage_embedding.models.crl.pseudo_labeling import compute_dataset_ublb


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

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')

# %%

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
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=100, transformations=None)

print(np.isnan(clrdat.full_dataset.images).any())
plt.imshow(clrdat.full_dataset.images[10])
plt.show()


# %%
colocs = ColocModel(clrdat)


# %%
model = CRL3(clrdat,
            num_cluster=36,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.13, # .8
            lower_iteration=0.24, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.18,
            pretraining_epochs=10,
            training_epochs=20, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            activation='sigmoid', # softmax
            clip_gradients=None
            )

 # %%
device='cuda'
model.train(logger=False)


# %% previous version
embds = model.inference_embeddings(new_data=model.data.test_dataset.images, normalize_images=False, device=device, use_embed_layer=True)
embds_ds = ds_coloc_convert(colocs=torch_cosine(torch.tensor(embds)),
                            ds_labels=model.data.test_dataset.dataset_labels,
                            ion_labels=model.data.test_dataset.ion_labels)


# %%
# Only taking DS latent similarities
cdsc = crl_ds_coloc(model, model.data.train_dataset, device=device)
crl_test_colocs, not_inferred = crl_latent_inference(cdsc, model.data.test_dataset.ion_labels, agg='mean')

# Aggregating across all latent similarities
# Currently performing much worse
# cdsc = crl_latent_coloc(model, model.data.train_dataset, device=device)
# crl_test_colocs, not_inferred = crl_fulllatent_coloc(cdsc, model.data.test_dataset.ion_labels, agg='median')


print('Not inferred:')
print(f'- Coloc: {round(colocs.test_mean_ni, 3)}')
print(f'- CRL:   {round(not_inferred, 3)}')


# %%
df_list = []

for quant in np.linspace(0.25, 0.45, 21):

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
    
    crltesteval = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                    test_dsl=colocs.data.test_dataset.dataset_labels,
                                    test_colocs=crl_test_colocs,
                                    upper_quantile=1.-quant, lower_quantile=quant)
    
    # crltesteval = colocs.quantile_gt(full_colocs=embds_ds,
    #                                  test_dsl=model.data.test_dataset.dataset_labels,
    #                                  test_il=model.data.test_dataset.ion_labels,
    #                                  upper_quantile=1.-quant, lower_quantile=quant)

    evaluation_dict = {'ground_truth': colocgt, 'predictions': {'crl': crltesteval, 
                                                                'colocmean': coloctesteval, 
                                                                'colocmedian': colocmed}, 
                       'uq': 1-quant, 'lq': quant}
    tmp = evaluation_quantile_overlap(evaluation_dict)

    df_list.append(tmp)
    

# %%
df = pd.concat(df_list).groupby(['model', 'lq', 'uq']).agg('mean').reset_index()

ax = sns.scatterplot(data=df, x='precision', y='recall', hue='model', size='uq')
ax.axline((1, 1), slope=1)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
plt.show()
sns.scatterplot(data=df, x='accuracy', y='f1score', hue='model', size='uq')



# %%
quant=.25
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

crltesteval = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                test_dsl=colocs.data.test_dataset.dataset_labels,
                                test_colocs=crl_test_colocs,
                                upper_quantile=1.-quant, lower_quantile=quant)

crltesteval = colocs.quantile_gt(full_colocs=embds_ds,
                                 test_dsl=model.data.test_dataset.dataset_labels,
                                 test_il=model.data.test_dataset.ion_labels,
                                 upper_quantile=1.-quant, lower_quantile=quant)

evaluation_dict = {'ground_truth': colocgt, 'predictions': {'crl': crltesteval, 
                                                            'colocmean': coloctesteval, 
                                                            'colocmedian': colocmed}, 
                    'uq': 1-quant, 'lq': quant}
tmp = evaluation_quantile_overlap(evaluation_dict)

sns.boxplot(data=tmp, x='model', y='accuracy')












# %%
imgs = torch.tensor(model.data.train_dataset.images[[1, 3, 21]])
imgs = imgs.reshape((imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]))

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.imshow(imgs[0][0])
ax2.imshow(imgs[1][0])
ax3.imshow(imgs[2][0])

a = model.inference_embeddings(imgs, normalize_images=False, normalize_embeddings=True, use_embed_layer=False)

print(torch_cosine(torch.tensor(a)))

print(torch_cosine(imgs.reshape((3, -1))))





# %%
feats = torch.Tensor([
            [1., 1., 0., 0.],
            [1., .9, .0, .0],
            [.1, .1, .1, .1],
            [.5, .1, .9, .8],
            [.6, .6, .6, .6],
            [.1, .2, .3, .4],
            [.4, .3, .2, .1],
            [.5, .5, .5, .5],
            [.8, .1, .2, .7],
            [.9, .0, .3, .9],
            ])
        
index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ds_labels = torch.tensor([0,0,1,1,1,1,0,0,0,0])
ions = torch.tensor([0,1,0,3,4,5,6,7,8,9])
ion_label_mat = torch.tensor(pairwise_same_elements(ions).astype(int))

crl = model.crl

sim_mat = crl.compute_ublb(feats)
        
# Compute cosine between all input images
gt_cosine = torch_cosine(torch.rand(10, 30))

dataset_ub, dataset_lb = compute_dataset_ublb(gt_cosine, ds_labels=ds_labels,
                                              lower_bound=20, upper_bound=80, device='cpu')

# Only for those values the loss will be evaluated
ds_mask = torch.zeros(sim_mat.shape, device='cpu')

# Loop over all datasets
for ds in torch.unique(ds_labels):
    ds_v = ds_labels == ds
    
    # Mask to subset similarities just to one dataset
    mask = torch.outer(ds_v, ds_v)
    mask2 = torch.eye(ds_mask.size(0), dtype=torch.bool)
    mask[mask2] = 0.

    # Set maskin with datasets to 1
    ds_mask[mask] = 1

# Align the same ions
ion_submat = ion_label_mat[index, :][:, index]

# Set same ions to 1 in target
ds_mask = torch.maximum(ds_mask, ion_submat)
gt_cosine[ion_submat==1.] = 1
# %%
torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)



# %%
import torch.nn.functional as functional

edges = torch.nonzero(pos_loc)
features = functional.normalize(feats, p=2, dim=-1)
edge_list = [(int(edge[0]), int(edge[1])) for edge in edges]


for x, y in edge_list:

# Convert the indices to a list of tuples representing edges

# %%
