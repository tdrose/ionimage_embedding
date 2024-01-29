# %%
import matplotlib.pyplot as plt
from torch import embedding

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL
from ionimage_embedding.models import gnnDiscrete


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
                    lr=1e-3, training_epochs=150, lightning_device='gpu')


# %%
mylogger = model.train()

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%


# %%
model = gnnDiscrete(data=dat, latent_dims=10, 
                    encoding = 'learned', embedding_dims=10,
                    lr=1e-3, training_epochs=150, lightning_device='gpu')


# %%
mylogger = model.train()

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%