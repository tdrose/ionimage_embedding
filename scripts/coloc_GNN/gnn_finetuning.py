# %%
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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Literal

from ionimage_embedding.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
from ionimage_embedding.models import gnnDiscrete
import ionimage_embedding.evaluation.latent as latent
import ionimage_embedding.evaluation.utils_gnn as utils_gnn
import ionimage_embedding.evaluation.metrics as metrics
from ionimage_embedding.logger import DictLogger

# %%
import os
os.system('nvidia-smi')


# %%
# Hyperparameters
tk = 10 # best for reconstruction loss: 2
bk = 10 # best for reconstruction loss: 1
min_images = tk + bk + 10 # Default 6
top_acc = 3
encoding = 'onehot'
early_stopping_patience = 3
activation: Literal['softmax', 'relu', 'sigmoid', 'none'] = 'none'
num_layers = 2

RANDOM_NETWORK = False

# %%
# Data
dat = ColocNetData_discrete(KIDNEY_LARGE, test=10, val=5, # 10, 5
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=tk, bottom_k=bk, random_network=RANDOM_NETWORK
                    )

mylogger = DictLogger()


# %% Initial training
model = gnnDiscrete(data=dat, latent_dims=20, 
                        encoding = encoding, embedding_dims=10,
                        lr=1e-3, training_epochs=130, 
                        early_stopping_patience=early_stopping_patience,
                        lightning_device='gpu', loss='coloc',
                        activation=activation, num_layers=num_layers)


mylogger = model.train()

pred_mc, pred_fraction = utils_gnn.mean_coloc_test(dat)


pred_cu = latent.coloc_umap_gnn(dat, k=3, n_components=10)


coloc_cu = latent.latent_colocinference(pred_cu, utils_gnn.coloc_ion_labels(dat, dat._test_set))


pred_gnn_t = latent.latent_gnn(model, dat, graph='training')
coloc_gnn_t = latent.latent_colocinference(pred_gnn_t, utils_gnn.coloc_ion_labels(dat, dat._test_set))

scenario_l = []
acc_l = []
eval_l = []
frac_l = []

acc_l.append(metrics.coloc_top_acc_gnn(pred_mc, dat, pred_mc, top=top_acc))
scenario_l.append('Mean coloc')
eval_l.append('Available')
frac_l.append(pred_fraction)

avail, trans, fraction = metrics.coloc_top_acc_gnn(coloc_cu, dat, pred_mc, top=top_acc)
scenario_l.append('UMAP')
acc_l.append(avail)
eval_l.append('Available')
frac_l.append(fraction)
scenario_l.append('UMAP')
acc_l.append(trans)
eval_l.append('Transitivity')
frac_l.append(fraction)

avail, trans, fraction = metrics.coloc_top_acc_gnn(coloc_gnn_t, dat, pred_mc, top=top_acc)
scenario_l.append('GNN training')
acc_l.append(avail)
eval_l.append('Available')
frac_l.append(fraction)
scenario_l.append('GNN training')
acc_l.append(trans)
eval_l.append('Transitivity')
frac_l.append(fraction)

acc_l.append(metrics.coloc_top_acc_gnn_random(coloc_gnn_t, dat, pred_mc, top=top_acc))
scenario_l.append('Random')
eval_l.append('Available')
frac_l.append(fraction)


# %% Performance
df = pd.DataFrame({'Scenario': scenario_l, 'Accuracy': acc_l, 
                   'Evaluation': eval_l, 'Fraction': frac_l})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Available'], x='Scenario', y='Accuracy', ax=ax1)
ax1.set_title(f'KIDNEY_LARGE (top-k: {tk}, bottom-k: {bk}, encoding: {encoding})')
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Available)')
ax1.set_ylim(0, 1)

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Scenario', y='Accuracy', ax=ax2)
ax2.set_title('Mean transifivity fraction: {:.2f}'.format(1-df[df['Evaluation']=='Transitivity']['Fraction'].mean()))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'Activation: {activation}')
plt.show()

plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()



# %% finetuning data
dat_finetune = dat.updated_k_data(top_k=9, bottom_k=9)

# %% finetuning

mylogger = model.fine_tune(dat_finetune, training_epochs=50)

pred_mc, pred_fraction = utils_gnn.mean_coloc_test(dat)


pred_cu = latent.coloc_umap_gnn(dat, k=3, n_components=10)


coloc_cu = latent.latent_colocinference(pred_cu, utils_gnn.coloc_ion_labels(dat, dat._test_set))


pred_gnn_t = latent.latent_gnn(model, dat, graph='training')
coloc_gnn_t = latent.latent_colocinference(pred_gnn_t, utils_gnn.coloc_ion_labels(dat, dat._test_set))

scenario_l = []
acc_l = []
eval_l = []
frac_l = []

acc_l.append(metrics.coloc_top_acc_gnn(pred_mc, dat, pred_mc, top=top_acc))
scenario_l.append('Mean coloc')
eval_l.append('Available')
frac_l.append(pred_fraction)

avail, trans, fraction = metrics.coloc_top_acc_gnn(coloc_cu, dat, pred_mc, top=top_acc)
scenario_l.append('UMAP')
acc_l.append(avail)
eval_l.append('Available')
frac_l.append(fraction)
scenario_l.append('UMAP')
acc_l.append(trans)
eval_l.append('Transitivity')
frac_l.append(fraction)

avail, trans, fraction = metrics.coloc_top_acc_gnn(coloc_gnn_t, dat, pred_mc, top=top_acc)
scenario_l.append('GNN training')
acc_l.append(avail)
eval_l.append('Available')
frac_l.append(fraction)
scenario_l.append('GNN training')
acc_l.append(trans)
eval_l.append('Transitivity')
frac_l.append(fraction)

acc_l.append(metrics.coloc_top_acc_gnn(coloc_gnn_t, dat, pred_mc, top=top_acc))
scenario_l.append('Random')
eval_l.append('Available')
frac_l.append(fraction)

# %% Performance
df = pd.DataFrame({'Scenario': scenario_l, 'Accuracy': acc_l, 
                   'Evaluation': eval_l, 'Fraction': frac_l})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Available'], x='Scenario', y='Accuracy', ax=ax1)
ax1.set_title(f'KIDNEY_LARGE (top-k: {tk}, bottom-k: {bk}, encoding: {encoding})')
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Available)')
ax1.set_ylim(0, 1)

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Scenario', y='Accuracy', ax=ax2)
ax2.set_title('Mean transifivity fraction: {:.2f}'.format(1-df[df['Evaluation']=='Transitivity']['Fraction'].mean()))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'Activation: {activation}')
plt.show()

plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%
