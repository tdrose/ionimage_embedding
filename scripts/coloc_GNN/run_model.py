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

from ionimage_embedding.dataloader.constants import CACHE_FOLDER
from ionimage_embedding.dataloader.ColocNet_data import ColocNetData_discrete
from ionimage_embedding.datasets import KIDNEY_SMALL, KIDNEY_LARGE
from ionimage_embedding.models import gnnDiscrete
from ionimage_embedding.evaluation.scoring import latent_colocinference
from ionimage_embedding.evaluation.gnn import (
    closest_accuracy_aggcoloc_ds, 
    closest_accuracy_latent_ds, 
    closest_accuracy_random_ds,
    mean_coloc_test,
    coloc_umap_ds,
    latent_gnn,
    coloc_ion_labels
    )
from ionimage_embedding.logger import DictLogger




# %%
import os
os.system('nvidia-smi')


# %%
# Hyperparameters
# #####################
# Hyperparameters
# #####################
# min_images
min_images = 30
# test
test = 5
# val
val = 3
# accuracy top-k
top_acc = 7
# Dataset
DSID = KIDNEY_LARGE
# Number of bootstraps
N_BOOTSTRAPS = 100


hyperparams_avail = {
    'latent_size': 27,
    'top_k': 2,
    'bottom_k':  2,
    'encoding': 'onehot', # 'recon', 'coloc'
    'early_stopping_patience': 2,
    'gnn_layer_type': 'GATv2Conv', # 'GCNConv', 'GATv2Conv', 'GraphConv'
    'loss_type': 'coloc',
    'num_layers': 1,
    'lr': 0.005945,
    'activation': 'none' # 'softmax', 'relu', 'sigmoid', 'none'
}

hyperparams_transitivity = {
    'latent_size': 23,
    'top_k': 3,
    'bottom_k':  7,
    'encoding': 'onehot', # 'recon', 'coloc'
    'early_stopping_patience': 10,
    'gnn_layer_type': 'GCNConv', # 'GCNConv', 'GATv2Conv', 'GraphConv'
    'loss_type': 'coloc',
    'num_layers': 1,
    'lr': 0.009140,
    'activation': 'none' # 'softmax', 'relu', 'sigmoid', 'none'
}


hyperparams = hyperparams_transitivity


scenario_l = []
acc_l = []
eval_l = []
frac_l = []

RANDOM_NETWORK = False

# %%
# Data
dat = ColocNetData_discrete(KIDNEY_LARGE, test=test, val=val, 
                    cache_images=True, cache_folder=CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
                    random_network=RANDOM_NETWORK
                    )

mylogger = DictLogger()

# %%
# Training
for i in range(N_BOOTSTRAPS):
    print('# #######')
    print(f'# Iteration {i}')
    print('# #######')
    
    dat.sample_sets()

    model = gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                        encoding = hyperparams['encoding'], embedding_dims=40,
                        lr=hyperparams['lr'], training_epochs=130, 
                        early_stopping_patience=hyperparams['early_stopping_patience'],
                        lightning_device='gpu', loss=hyperparams['loss_type'],
                        activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                        gnn_layer_type=hyperparams['gnn_layer_type'])


    mylogger = model.train()

    pred_mc, pred_fraction = mean_coloc_test(dat)
    

    pred_cu = coloc_umap_ds(dat, k=3, n_components=10)
    

    coloc_cu = latent_colocinference(pred_cu, coloc_ion_labels(dat, dat._test_set))

    
    pred_gnn_t = latent_gnn(model, dat, graph='training')
    coloc_gnn_t = latent_colocinference(pred_gnn_t, coloc_ion_labels(dat, dat._test_set))
    
    
    acc_l.append(closest_accuracy_aggcoloc_ds(pred_mc, dat, top=top_acc))
    scenario_l.append('Mean coloc')
    eval_l.append('Available')
    frac_l.append(pred_fraction)

    avail, trans, fraction = closest_accuracy_latent_ds(coloc_cu, dat, pred_mc, top=top_acc)
    scenario_l.append('UMAP')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('UMAP')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    avail, trans, fraction = closest_accuracy_latent_ds(coloc_gnn_t, dat, pred_mc, top=top_acc)
    scenario_l.append('GNN training')
    acc_l.append(avail)
    eval_l.append('Available')
    frac_l.append(fraction)
    scenario_l.append('GNN training')
    acc_l.append(trans)
    eval_l.append('Transitivity')
    frac_l.append(fraction)

    acc_l.append(closest_accuracy_random_ds(coloc_gnn_t, dat, top=top_acc))
    scenario_l.append('Random')
    eval_l.append('Available')
    frac_l.append(fraction)
    

    #print(f'Fraction of predicted colocs: {pred_fraction:.2f}')

# %%
# Evaluation
df = pd.DataFrame({'Scenario': scenario_l, 'Accuracy': acc_l, 
                   'Evaluation': eval_l, 'Fraction': frac_l})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Available'], x='Scenario', y='Accuracy', ax=ax1)
ax1.set_title('KIDNEY_LARGE (top-k: {}, bottom-k: {}, encoding: {})'.format(hyperparams['top_k'],
                                                                            hyperparams['bottom_k'],
                                                                            hyperparams['encoding']))
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Available)')
ax1.set_ylim(0, 1)

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Scenario', y='Accuracy', ax=ax2)
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(1-df[df['Evaluation']=='Transitivity']['Fraction'].mean()))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle('Activation: {}'.format(hyperparams['activation']))

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%
