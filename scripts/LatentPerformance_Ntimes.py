# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
from typing import Dict, Union
from typing import List
import pandas as pd

from ionimage_embedding.models import CRL, ColocModel, BioMedCLIP
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
# prerequisites

Ntimes = 10
filename = 'LatentPerformance_Ntimes.csv'
topN = 5
coloc_agg='mean'
device='cuda'

dataset_name = 'KIDNEY_SMALL'
dataset = KIDNEY_SMALL

performance_dict: Dict[str, List[Union[float, str]]] = {'model': [],
                                                        'iteration': [],
                                                        'dataset': [],
                                                        'scenario': [],
                                                        'topN_accuracy': []}

# %% InfoNCE
for i in range(Ntimes):
    lt = 'infoNCE'
    print(f'{lt} {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=50, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=False, knn=True, lr=0.08)
    
    colocs = ColocModel(iid)

    mylogger = model.train(logger=True)

    latent_model = latent_centroids_df(model, origin='train')
    model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)
    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i+1)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(model_coloc_inferred, 
                                                                     colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% selfContrast
for i in range(Ntimes):
    lt = 'selfContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = ColocModel(iid)

    mylogger = model.train(logger=True)

    latent_model = latent_centroids_df(model, origin='train')
    model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)
    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(model_coloc_inferred, 
                                                                     colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% colocContrast
for i in range(Ntimes):
    lt = 'colocContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = ColocModel(iid)

    mylogger = model.train(logger=True)

    latent_model = latent_centroids_df(model, origin='train')
    model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)
    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(model_coloc_inferred, 
                                                                     colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% regContrast
for i in range(Ntimes):
    lt = 'regContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = ColocModel(iid)

    mylogger = model.train(logger=True)

    latent_model = latent_centroids_df(model, origin='train')
    model_coloc_inferred = latent_colocinference(latent_model, colocs.data.test_dataset.ion_labels)
    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(model_coloc_inferred, 
                                                                     colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% BMC
for i in range(Ntimes):
    lt = 'BMC'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    
    colocs = ColocModel(iid)

    bmc = BioMedCLIP(data=iid)

    latent_bmc = latent_centroids_df(bmc, origin='train')
    bmc_coloc_inferred = latent_colocinference(latent_bmc, colocs.data.test_dataset.ion_labels)

    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(bmc_coloc_inferred, colocs, 
                                                                     top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% Mean coloc
for i in range(Ntimes):
    lt = 'Mean coloc'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    
    colocs = ColocModel(iid)    
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_aggcoloc(colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% UMAP coloc
for i in range(Ntimes):
    lt = 'Coloc UMAP'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    
    colocs = ColocModel(iid)

    coloc_embedding = coloc_umap(colocs, k=3, n_components=5)
    umap_coloc_inferred = latent_colocinference(coloc_embedding, colocs.data.test_dataset.ion_labels)
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_latent(umap_coloc_inferred, colocs, top=topN))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)


# %% Random
for i in range(Ntimes):
    lt = 'Random'
    print(f'{lt}: {i}/{Ntimes}')

    iid = IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    
    colocs = ColocModel(iid)

    bmc = BioMedCLIP(data=iid)
    dsc_dict = compute_ds_coloc(bmc, origin='test')
    
    performance_dict['model'].append(lt)
    performance_dict['iteration'].append(i)
    performance_dict['dataset'].append(dataset_name)
    performance_dict['scenario'].append('random')
    performance_dict['topN_accuracy'].append(closest_accuracy_random(dsc_dict, colocs, top=topN, origin='test'))
    
    pd.DataFrame(performance_dict).to_csv(filename, index=False)

# %% Plotting
import seaborn as sns

ax = sns.boxplot(data=pd.DataFrame(performance_dict), x='model', y='topN_accuracy')

# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
# %%
