# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from typing import Dict
import pandas as pd

from ionimage_embedding.models import CRL, BioMedCLIP
from ionimage_embedding import ColocModel
from ionimage_embedding.dataloader import IonImagedata_random
from ionimage_embedding.evaluation.scoring import (
    closest_accuracy_aggcoloc,
    closest_accuracy_latent,
    closest_accuracy_random,
    compute_ds_coloc,
    latent_dataset_silhouette,
    same_ion_similarity,
    coloc_umap,
    latent_colocinference,
    closest_accuracy_coloclatent,
    remove_nan,
    general_mse
)
from ionimage_embedding.evaluation.utils import cluster_latent, latent_centroids_df
from ionimage_embedding.datasets import KIDNEY_SMALL, BRAIN_SMALL
from ionimage_embedding.evaluation.plots import umap_latent,  umap_allorigin, plot_image_clusters


# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
    print('Running in IPython, auto-reload enabled!')
except ImportError:
    # Not in IPython, continue with normal Python code
    pass


# %%

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')


# %%

crldat = IonImagedata_random(KIDNEY_SMALL, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=60, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9, vitb16_compatible=True)

# print(np.isnan(crldat.full_dataset.images).any())
# plt.imshow(crldat.full_dataset.images[10])
# plt.show()

print('Train data:\t\t', crldat.train_dataset.images.shape)
print('Train data:\t\t', crldat.train_dataset.images.shape[0])
print('Validation data:\t', crldat.val_dataset.images.shape[0])
print('Test data:\t\t', crldat.test_dataset.images.shape[0])
# %%
colocs = ColocModel(crldat)



# %%
model = CRL(crldat,
            num_cluster=20,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.0, # .8
            lower_iteration=0.0, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.08, # .18
            pretraining_epochs=10,
            training_epochs=100, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            cnn_dropout=0.01,
            activation='relu', # softmax
            loss_type='infoNCE', # 'selfContrast', 'colocContrast', 'regContrast', 'infoNCE'
            architecture='resnet18', # 'resnet18
            resnet_pretrained=True,
            clip_gradients=None
            )

# %%
device='cuda'
mylogger = model.train(logger=True)

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()

# %%
ds = 'test'
top = 5
coloc_agg='mean'

# %%
bmc = BioMedCLIP(data=crldat)
bmc_dsc_dict = compute_ds_coloc(bmc, origin=ds) # type: ignore

# %%
coloc_embedding = coloc_umap(colocs, k=3, n_components=5)
umap_coloc_inferred = latent_colocinference(coloc_embedding, colocs.data.test_dataset.ion_labels)

latent_model = latent_centroids_df(model, origin='train')
model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)

latent_bmc = latent_centroids_df(bmc, origin='train')
bmc_coloc_inferred = latent_colocinference(latent_bmc, colocs.data.test_dataset.ion_labels)


# %%
dsc_dict = compute_ds_coloc(model, origin=ds)

# %%
if ds == 'test':
    print(f'{coloc_agg} accuracy: ', closest_accuracy_aggcoloc(colocs, top=top))
    print('Latent: ')
    print('* Coloc UMAP accuracy: ', closest_accuracy_latent(umap_coloc_inferred, colocs, top=top))
    print('* Model accuracy: ', closest_accuracy_latent(model_coloc_inferred, colocs, top=top))
    print('* BMC accuracy: ', closest_accuracy_latent(bmc_coloc_inferred, colocs, top=top))


print('Model accuracy: ', closest_accuracy_coloclatent(dsc_dict, colocs, top=top, origin=ds))
print('BMC accuracy: ', closest_accuracy_coloclatent(bmc_dsc_dict, colocs, top=top, origin=ds))
print('Random accuracy: ', closest_accuracy_random(dsc_dict, colocs, top=top, origin=ds))

print('Silhouette scores:')
print('* Model: ', latent_dataset_silhouette(model, origin=ds))
print('* BMC: ', latent_dataset_silhouette(bmc, origin=ds))


# %%

model_colocs_filered = remove_nan(model_coloc_inferred)
bmc_colocs_filered = remove_nan(bmc_coloc_inferred)
umap_colocs_filered = remove_nan(umap_coloc_inferred)

# Subset coloc_test to only include colocs that are in the model
mean_colocs_filtered = colocs.test_mean_coloc.loc[model_colocs_filered.index,  # type: ignore
                                                  model_colocs_filered.columns]

print('Transitivity MSE:')
print('* Model: ', general_mse(model_colocs_filered, mean_colocs_filtered, colocs.test_coloc))
print('* BMC: ', general_mse(bmc_colocs_filered, mean_colocs_filtered, colocs.test_coloc))
print('* UMAP: ', general_mse(umap_colocs_filered, mean_colocs_filtered, colocs.test_coloc))
print('* Mean: ', general_mse(mean_colocs_filtered, mean_colocs_filtered, colocs.test_coloc))














# %%
umap_allorigin(model, device=device)
umap_latent(model, origin='train', device=device)








# %%
from typing import Literal
from sklearn.metrics import pairwise_kernels
import numpy as np
import pandas as pd
import seaborn as sns
from ionimage_embedding.evaluation.utils import get_latent, get_ion_labels



# %%
sim = same_ion_similarity(model, origin='test')
sns.violinplot(data=sim, x='type', y='similarity', cut=0)




# %%

from ionimage_embedding.evaluation.plots import compute_umap
from ionimage_embedding.evaluation.utils import get_ds_labels

# %%
ds = 'train'
cluster_df = cluster_latent(model, n_clusters=4, plot=True, origin=ds)
# %%

plot_image_clusters(model, cluster_df, origin=ds)
# %%
