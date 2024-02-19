# %%
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

import ionimage_embedding as iie

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
dat = iie.dataloader.ColocNet_data.ColocNetData_discrete(iie.datasets.KIDNEY_SMALL, test=2, val=1, 
                 cache_images=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=60, min_images=6, top_k=3,
                 maxzero=.9)





# %%
from torch_geometric.loader import DataLoader
dl = DataLoader(dat.dataset, batch_size=1, shuffle=True)

# %%
tmp = next(iter(dl))
