# %%
# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import upsetplot

from ionimage_embedding.evaluation.utils import get_ion_labels, get_latent
from ionimage_embedding.models import CRL, ColocModel
from ionimage_embedding.dataloader import CRLdata
from ionimage_embedding.evaluation.scoring import (
    closest_accuracy_aggcoloc,
    closest_accuracy_latent,
    closest_accuracy_random,
    compute_ds_coloc,
    latent_dataset_silhouette,
    same_ion_similarity,
    coloc_umap,
    umap_inference,
    closest_accuracy_umapcoloc
)
from ionimage_embedding.evaluation.utils import cluster_latent
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
except ImportError:
    # Not in IPython, continue with normal Python code
    pass



# %%
def ionlabel_hist(data: CRLdata, ax: plt.Axes):
    arr = data.full_dataset.ion_labels.detach().cpu().numpy()
    
    _, counts = np.unique(arr, return_counts=True)
    
    ax.plot(np.arange(len(counts)), np.sort(counts), '-')
    ax.set_title('Ion detection')
    ax.set_xlabel('Ion')
    ax.set_ylabel('#datasets')

def ions_per_dataset(data: CRLdata, ax: plt.Axes):
    arr = data.full_dataset.dataset_labels.detach().cpu().numpy()

    _, counts = np.unique(arr, return_counts=True)
    
    plt.plot(np.arange(len(counts)), np.sort(counts), '-')
    ax.set_title('Ions per dataset')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('#ions')

def ion_overview(data: CRLdata):
    fig, (ax0, ax1) = plt.subplots(ncols=2)

    ionlabel_hist(data, ax0)
    ions_per_dataset(data, ax1)

    plt.show()

def ion_upset(data: CRLdata):

    ion_arr = data.full_dataset.ion_labels.detach().cpu().numpy()
    ds_arr = data.full_dataset.dataset_labels.detach().cpu().numpy()

    ion_dict = {}
    for dsid in set(ds_arr):
        mask = ds_arr == dsid
        ion_dict[data.dsl_int_mapper[dsid]] = ion_arr[mask]

    upset_input = upsetplot.from_contents(ion_dict)

    upsetplot.UpSet(upset_input, subset_size="count").plot()

def example_images(data: CRLdata, ncols: int=5, figsize=(10, 5), titlesize=1):

    ds_arr = data.full_dataset.dataset_labels.detach().cpu().numpy()
    imgs = data.full_dataset.images
    n_ds = len(set(ds_arr))

    nrows = n_ds//ncols
    if n_ds%ncols > 0:
        nrows +=1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axs = axs.flatten()
    
    for dsid in set(ds_arr):
        mask = ds_arr == dsid
        curr_images = imgs[mask]
        id = np.random.choice(len(imgs), size=1)[0]

        axs[dsid].imshow(imgs[id])
        axs[dsid].axis('off')
        axs[dsid].set_title(data.dsl_int_mapper[dsid], size=titlesize)
    
    if n_ds%ncols > 0:
        for i in range(-ncols+(n_ds%ncols), 0):
            axs[i].axis('off')

    plt.plot()   




# %% KIDNEY SMALL
ds_list = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s',
    '2022-12-07_01h02m53s',
    '2022-12-07_01h01m06s',
    # '2022-11-28_22h24m25s',
    '2022-11-28_22h23m30s'
                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)

# %% KIDNEY LARGE
ds_list = [
    '2022-12-07_02h13m50s', '2020-01-13_23h01m15s', '2020-01-09_18h27m19s',
    '2020-01-09_18h42m02s', '2020-01-09_19h06m59s', '2020-01-09_20h17m08s',
    '2020-01-09_21h03m05s', '2022-12-07_02h13m20s', '2020-01-13_22h56m28s',
    '2020-01-13_22h57m18s', '2020-01-14_23h33m56s', '2020-01-06_23h13m42s',
    '2020-01-15_23h06m55s', '2020-01-21_21h26m03s', '2020-01-21_22h10m23s',
    '2020-01-21_22h56m53s',
    '2020-03-11_16h38m34s', '2020-03-11_16h40m10s', '2020-01-06_23h24m27s',
    '2020-01-06_23h13m06s', '2020-12-09_02h36m29s', '2018-09-07_00h49m45s',
    '2018-06-29_05h34m01s', '2018-06-29_05h38m12s', '2018-06-29_23h47m27s',
    '2018-09-07_00h24m18s', '2018-09-07_00h36m53s', '2018-09-13_00h55m36s',
    '2018-09-13_00h55m56s', '2018-09-14_18h15m42s', '2018-09-07_00h52m21s',
    '2020-01-06_22h04m12s', '2018-09-07_01h02m31s', '2018-11-21_21h31m33s',
    '2018-11-21_21h36m35s', '2019-03-06_21h42m13s', '2019-07-19_19h41m43s',
    '2019-07-23_23h36m29s', '2020-01-06_22h02m51s', '2020-01-06_22h03m31s',
    '2020-12-03_19h22m16s', '2020-01-09_21h14m40s', '2022-11-28_22h24m25s',
    '2021-03-04_23h29m12s', '2022-12-07_02h09m41s', '2022-12-07_02h10m45s',
    '2022-12-07_01h01m06s', '2021-03-04_23h29m53s', '2021-03-04_23h31m12s',
    '2022-12-07_02h08m52s', '2020-12-09_02h41m05s'
                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

# %%
ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9, figsize=(20, 10), titlesize=5)

# %% BRAIN SMALL
ds_list = [
    '2021-11-11_11h49m37s', '2022-05-31_10h27m17s',
    '2022-05-30_20h44m19s', '2022-01-31_09h08m54s', '2022-01-31_08h54m51s',
    '2017-02-17_14h56m37s', '2017-02-17_14h41m43s',
    '2022-08-23_23h48m59s', '2018-08-01_14h25m56s', '2018-08-01_14h26m17s',
    '2018-08-01_14h25m21s', '2016-09-21_16h06m55s', '2018-08-01_14h21m22s',
    '2022-07-19_19h29m24s', '2016-09-21_16h06m56s'
                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9, figsize=(20, 10), titlesize=5)

# %% BRAIN LARGE
ds_list = [

                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)


# %% LUNG SMALL
ds_list = [

                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)

# %% LUNG LARGE
ds_list = [

                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)


# %% LIVER SMALL
ds_list = [

                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)

# %% LIVER LARGE
ds_list = [

                  ]

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/tmp/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

ion_overview(crldat)
ion_upset(crldat)
example_images(crldat, ncols=9)