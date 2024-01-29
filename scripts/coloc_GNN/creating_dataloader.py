# %%
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL


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
from torch_geometric.loader import DataLoader
dl = DataLoader(dat.dataset, batch_size=1, shuffle=True)

# %%
tmp = next(iter(dl))








# %%

dataset = MoleculeNet(root=CACHE_FOLDER, name="Tox21")

# %%
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# %%
tmp = next(iter(train_loader))
print(tmp)
print(f'Batch size: {len(tmp.y)}')
print(f'X shape: {tmp.x.shape}')
print(f'Edge index shape: {tmp.edge_index.shape}')

# %%

tmp.x




# %%
