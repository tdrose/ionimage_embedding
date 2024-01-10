# %%
import torchvision.transforms as T


from ionimage_embedding.dataloader import CRLdata
from ionimage_embedding.models import ColocModel, BioMedCLIP
from ionimage_embedding.datasets import KIDNEY_SMALL
from ionimage_embedding.evaluation.scoring import (
    closest_accuracy_aggcoloc,
    closest_accuracy_latent,
    closest_accuracy_random,
    compute_ds_coloc
)

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

crldat = CRLdata(KIDNEY_SMALL, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=40, 
                 transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)


# %%
bmc = BioMedCLIP(data=crldat)

# %%
colocs = ColocModel(crldat)


# %%
ds = 'test'
top = 3
coloc_agg='mean'
dsc_dict = compute_ds_coloc(bmc, origin=ds) # type: ignore


print('Model accuracy: ', closest_accuracy_latent(dsc_dict, colocs, top=top, origin=ds))
print('Random accuracy: ', closest_accuracy_random(dsc_dict, colocs, top=top, origin=ds))
if ds == 'test':
    print(f'{coloc_agg} accuracy: ', closest_accuracy_aggcoloc(colocs, top=top))
