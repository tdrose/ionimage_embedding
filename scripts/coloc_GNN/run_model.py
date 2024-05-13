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

import ionimage_embedding as iie


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
DSID = iie.datasets.KIDNEY_LARGE
DS_NAME = 'KIDNEY_LARGE'
# Number of bootstraps
N_BOOTSTRAPS = 10


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

hyperparams_transitivity2 = {
    'latent_size': 22,
    'top_k': 15,
    'bottom_k':  15,
    'encoding': 'onehot', # 'recon', 'coloc'
    'early_stopping_patience': 10,
    'gnn_layer_type': 'GATv2Conv', # 'GCNConv', 'GATv2Conv', 'GraphConv'
    'loss_type': 'coloc',
    'num_layers': 1,
    'lr': 0.006479,
    'activation': 'none' # 'softmax', 'relu', 'sigmoid', 'none'
}


hyperparams = hyperparams_transitivity


RANDOM_NETWORK = False

# %%
# Data
dat = iie.dataloader.ColocNet_data.ColocNetData_discrete(
    DSID, test=test, val=val, 
    cache_images=True, cache_folder=iie.constants.CACHE_FOLDER,
    colocml_preprocessing=True, 
    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
    top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
    random_network=RANDOM_NETWORK
                    )

mylogger = iie.logger.DictLogger()


# %%
    
# acc_perf = iie.logger.PerformanceLogger('Model','Accuracy', 
#                                         'Evaluation', 'Fraction')
# mse_perf = iie.logger.PerformanceLogger('Model','MSE',
#                                         'Evaluation', 'Fraction')

perf_df: pd.DataFrame | None = None

# Training
for i in range(N_BOOTSTRAPS):
    print('# #######')
    print(f'# Iteration {i}')
    print('# #######')
    
    dat.sample_sets()

    model = iie.models.gnn.gnnd.gnnDiscrete(
        data=dat, latent_dims=hyperparams['latent_size'], 
        encoding = hyperparams['encoding'], embedding_dims=40,
        lr=hyperparams['lr'], training_epochs=130, 
        early_stopping_patience=hyperparams['early_stopping_patience'],
        lightning_device='gpu', loss=hyperparams['loss_type'],
        activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
        gnn_layer_type=hyperparams['gnn_layer_type'])


    mylogger = model.train()

    pred_mc, pred_fraction = iie.evaluation.utils_gnn.mean_coloc_test(dat)
    
    pred_cu = iie.evaluation.latent.coloc_umap_gnn(dat, k=3, n_components=10)
    coloc_cu = iie.evaluation.latent.latent_colocinference(pred_cu, 
                                     iie.evaluation.utils_gnn.coloc_ion_labels(dat, dat._test_set))
    
    pred_gnn_t = iie.evaluation.latent.latent_gnn(model, dat, graph='training')
    coloc_gnn_t = iie.evaluation.latent.latent_colocinference(
        pred_gnn_t, 
        iie.evaluation.utils_gnn.coloc_ion_labels(dat, dat._test_set))
    
    # Log performance metrics
    perf_df = iie.evaluation.metrics.evaluation_gnn(perf_df, iie.constants.MEAN_COLOC,
                                                    pred_mc, dat, pred_mc, top_acc=top_acc)
    
    perf_df = iie.evaluation.metrics.evaluation_gnn(perf_df, iie.constants.UMAP,
                                                    coloc_cu, dat, pred_mc, top_acc=top_acc)
    
    perf_df = iie.evaluation.metrics.evaluation_gnn(perf_df, iie.constants.GNN,
                                                    coloc_gnn_t, dat, pred_mc, top_acc=top_acc)
    
    perf_df = iie.evaluation.metrics.evaluation_gnn(perf_df, iie.constants.RANDOM,
                                                    coloc_gnn_t, dat, pred_mc, top_acc=top_acc, 
                                                    random=True)
    
    
    # Accuracy
    # avail, trans, _ = iie.evaluation.metrics.coloc_top_acc_gnn(pred_mc, dat, 
    #                                                                   pred_mc, top=top_acc)
    # acc_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 1)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_gnn(coloc_cu, dat, 
    #                                                                   pred_mc, top=top_acc)
    # acc_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction)
    # acc_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_gnn(coloc_gnn_t, dat, 
    #                                                                   pred_mc, top=top_acc)
    # acc_perf.add_result(iie.constants.GNN, avail, 'Co-detected', fraction)
    # acc_perf.add_result(iie.constants.GNN, trans, 'Transitivity', 1-fraction)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_gnn_random(pred_mc, dat, 
    #                                                                          pred_mc, top=top_acc) 
    # acc_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction)
    # acc_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction)
    
    # # MSE
    # avail, trans, fraction = iie.evaluation.metrics.coloc_mse_gnn(pred_mc, pred_mc, dat)
    # mse_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 1)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_mse_gnn(coloc_cu, pred_mc, dat)
    # mse_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction)
    # mse_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_mse_gnn(coloc_gnn_t, pred_mc, dat)
    # mse_perf.add_result(iie.constants.GNN, avail, 'Co-detected', fraction)
    # mse_perf.add_result(iie.constants.GNN, trans, 'Transitivity', 1-fraction)

    # avail, trans, fraction = iie.evaluation.metrics.coloc_mse_gnn_random(coloc_gnn_t, pred_mc, dat)
    # mse_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction)
    # mse_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction)


# %% Plotting parameter

XXSMALLER_SIZE = 5
XSMALLER_SIZE = 6
SMALLER_SIZE = 8
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
XBIGGER_SIZE = 20
XXBIGGER_SIZE = 28

cm = 1/2.54


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=XBIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=XBIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=XBIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=XBIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=XBIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=XBIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'text.color': "595a5cff",
                     'axes.labelcolor': "595a5cff",
                     'xtick.color': "595a5cff",
                     'ytick.color': "595a5cff",})

my_pal = iie.constants.MODEL_PALLETE



# %%
# Accuracy 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.violinplot(data=perf_df, x='model', y='co-detection accuracy', ax=ax1, 
               palette=my_pal, cut=0)
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
ax1.set_ylim(0, 1)

sns.violinplot(data=perf_df, x='model', y='transitivity accuracy', ax=ax2,
               palette=my_pal, cut=0)
frac = perf_df['nan_fraction'].mean()
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(frac))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'{DS_NAME}, Leave out datasets')
plt.show()

# %%
# MSE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.violinplot(data=perf_df, x='model', y='co-detection mse', ax=ax1,
               palette=my_pal, cut=0)
sns.despine(offset=5, trim=False, ax=ax1)
ax1.set_ylabel(f'MSE (Co-detected)')

sns.violinplot(data=perf_df, x='model', y='transitivity mse', ax=ax2,
               palette=my_pal, cut=0)
frac = perf_df['nan_fraction'].mean()
ax2.set_title('Transitivity fraction: {:.1%}'.format(frac))
ax2.set_ylabel(f'MSE (Transitivity)')
sns.despine(offset=5, trim=False, ax=ax2)

#plt.show()
# plt.savefig('/g/alexandr/tim/tmp/mse_df_fig.pdf', bbox_inches='tight')

# %%
# MAE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.violinplot(data=perf_df, x='model', y='co-detection mae', ax=ax1,
               palette=my_pal, cut=0)
sns.despine(offset=5, trim=False, ax=ax1)
ax1.set_ylabel(f'MAE (Co-detected)')

sns.violinplot(data=perf_df, x='model', y='transitivity mae', ax=ax2,
               palette=my_pal, cut=0)
frac = perf_df['nan_fraction'].mean()
ax2.set_title('Transitivity fraction: {:.1%}'.format(frac))
ax2.set_ylabel(f'MAE (Transitivity)')
sns.despine(offset=5, trim=False, ax=ax2)

# %%
# SMAPE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

sns.violinplot(data=perf_df, x='model', y='co-detection smape', ax=ax1,
               palette=my_pal, cut=0)
sns.despine(offset=5, trim=False, ax=ax1)
ax1.set_ylabel(f'SMAPE (Co-detected)')

sns.violinplot(data=perf_df, x='model', y='transitivity smape', ax=ax2,
               palette=my_pal, cut=0)
frac = perf_df['nan_fraction'].mean()
ax2.set_title('Transitivity fraction: {:.1%}'.format(frac))
ax2.set_ylabel(f'SMAPE (Transitivity)')
sns.despine(offset=5, trim=False, ax=ax2)

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()
# %%
