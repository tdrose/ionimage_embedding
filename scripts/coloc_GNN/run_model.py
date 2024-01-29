# %%
import matplotlib.pyplot as plt
import torch
import numpy as np

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding import ColocModel

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
dat = ColocNetData_discrete(KIDNEY_SMALL, test=2, val=1, 
                 cache_images=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=60, min_images=6, top_k=3,
                 maxzero=.9)



# %%
model = gnnDiscrete(data=dat, latent_dims=10, 
                    encoding = 'onehot',
                    lr=1e-3, training_epochs=170, lightning_device='gpu')


# %%
mylogger = model.train()

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()



# %%
def mean_colocs(data: ColocNetData_discrete):
    
    train_ds = data._train_set

    test_ds = data._test_set
    
    test_ions = []
    for dsid in test_ds:
        test_ions.extend(data.dataset.coloc_dict[dsid].columns)
    test_ions = torch.tensor(list(set(test_ions)))
    # Compute mean coloc across training datasets
    mean_colocs = ColocModel._inference(ds_colocs={k: v for k, v in data.dataset.coloc_dict.items() if k in train_ds}, 
                                        ion_labels=test_ions, agg='mean')
    
    return mean_colocs

tmp = mean_colocs(dat)
# %%
