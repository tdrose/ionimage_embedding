# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from typing import Dict
import pandas as pd

from ionimage_embedding.models import CVAE, BioMedCLIP, ColocModel
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
from ionimage_embedding.datasets import KIDNEY_SMALL
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
import os
os.system('nvidia-smi')

# %%
imagedat = IonImagedata_random(KIDNEY_SMALL, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9, vitb16_compatible=True)

print('Train data:\t\t', imagedat.train_dataset.images.shape)
print('Train data:\t\t', imagedat.train_dataset.images.shape[0])
print('Validation data:\t', imagedat.val_dataset.images.shape[0])
print('Test data:\t\t', imagedat.test_dataset.images.shape[0])

# %%
model = CVAE(data=imagedat, 
             num_cluster=20, 
             lr=0.001, 
             training_epochs=11, 
             lightning_device='gpu',
             clip_gradients=None,
             activation='relu', # softmax
             weight_decay=1e-4)

# %%
device='cuda'
mylogger = model.train()


# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()

# %%
colocs = ColocModel(imagedat)

# %%
ds = 'test'
top = 5
coloc_agg='mean'

# %%
bmc = BioMedCLIP(data=imagedat)
bmc_dsc_dict = compute_ds_coloc(bmc, origin=ds) # type: ignore

# %%
coloc_embedding = coloc_umap(colocs, k=3, n_components=5)
umap_coloc_inferred = latent_colocinference(coloc_embedding, colocs.data.test_dataset.ion_labels)

latent_model = latent_centroids_df(model, origin='train')
model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)

latent_bmc = latent_centroids_df(bmc, origin='train')
bmc_coloc_inferred = latent_colocinference(latent_bmc, colocs.data.test_dataset.ion_labels)

if ds == 'test':
    print(f'{coloc_agg} accuracy: ', closest_accuracy_aggcoloc(colocs, top=top))
    print('Latent: ')
    print('* Coloc UMAP accuracy: ', closest_accuracy_latent(umap_coloc_inferred, colocs, top=top))
    print('* Model accuracy: ', closest_accuracy_latent(model_coloc_inferred, colocs, top=top))
    print('* BMC accuracy: ', closest_accuracy_latent(bmc_coloc_inferred, colocs, top=top))

# %%
print('Silhouette scores:')
print('* Model: ', latent_dataset_silhouette(model, origin=ds))
print('* BMC: ', latent_dataset_silhouette(bmc, origin=ds))