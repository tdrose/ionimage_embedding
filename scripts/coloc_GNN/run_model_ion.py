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
min_images = 11
# test
test = 1
# val
val = 1
# accuracy top-k
top_acc = 3
# Dataset
DSID = iie.datasets.KIDNEY_LARGE
DS_NAME = 'KIDNEY_LARGE'
# Number of bootstraps
N_BOOTSTRAPS = 50


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
acc_perf = iie.logger.PerformanceLogger('Model','Accuracy', 
                                        'Evaluation', 'Fraction')
mse_perf = iie.logger.PerformanceLogger('Model', 'MSE',
                                        'Evaluation', 'Fraction')

# II Data
iidata = iie.dataloader.IonImage_data.IonImagedata_random(
    DSID, test=.1, val=.1, transformations=None, fdr=.1,
    min_images=min_images, maxzero=.9, batch_size=10, 
    colocml_preprocessing=True, cache=True)

# %%
for i in range(N_BOOTSTRAPS):
    print('# #######')
    print(f'# Iteration {i}')
    print('# #######')

    iidata.sample_sets()

    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iidata, device='cuda')

    # ColocNet Data
    dat = iie.dataloader.ColocNet_data.ColocNetData_discrete([""], 
                        test=test, val=val, 
                        cache_images=True, cache_folder=iie.constants.CACHE_FOLDER,
                        colocml_preprocessing=True, 
                        fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                        top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
                        random_network=RANDOM_NETWORK,
                        use_precomputed=True,
                        ds_labels=iidata.train_dataset.dataset_labels,
                        ion_labels=iidata.train_dataset.ion_labels,
                        coloc=colocs.train_coloc,
                        dsl_int_mapper=iidata.dsl_int_mapper,
                        ion_int_mapper=iidata.ion_int_mapper,
                        n_ions=int(iidata.full_dataset.ion_labels.max().numpy())+1,
                        force_reload=True,
                        )

    mylogger = iie.logger.DictLogger()

    # Define model
    model = iie.models.gnn.gnnd.gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                            encoding = hyperparams['encoding'], embedding_dims=40,
                            lr=hyperparams['lr'], training_epochs=130, 
                            early_stopping_patience=hyperparams['early_stopping_patience'],
                            lightning_device='gpu', loss=hyperparams['loss_type'],
                            activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                            gnn_layer_type=hyperparams['gnn_layer_type'])

    mylogger = model.train()

    pred_mc = colocs.test_mean_coloc

    coloc_embedding = iie.evaluation.latent.coloc_umap_iid(colocs, k=3, n_components=5)
    coloc_cu = iie.evaluation.latent.latent_colocinference(
        coloc_embedding, 
        colocs.test_dataset.ion_labels)
    
    pred_gnn_t = iie.evaluation.latent.latent_gnn(model, dat, graph='training')
    coloc_gnn_t = iie.evaluation.latent.latent_colocinference(pred_gnn_t, colocs.test_dataset.ion_labels)
    
    
    # Accuracy
    avail, trans, _ = iie.evaluation.metrics.coloc_top_acc_iid(latent=pred_mc,
                                                               agg_coloc_pred=pred_mc,
                                                               colocs=colocs, 
                                                               top=top_acc)
    acc_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 1)

    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=coloc_cu,
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction)
    acc_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction)

    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=coloc_gnn_t, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(iie.constants.GNN, avail, 'Co-detected', fraction)
    acc_perf.add_result(iie.constants.GNN, trans, 'Transitivity', 1-fraction)

    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid_random(pred_mc, 
                                                                             agg_coloc_pred=pred_mc,
                                                                             colocs=colocs, 
                                                                             top=top_acc) 
    acc_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction)
    acc_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction)
    
    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(pred_mc, pred_mc, colocs)
    mse_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 1)

    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(coloc_cu, pred_mc, colocs)
    mse_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction)
    mse_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction)

    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(coloc_gnn_t, pred_mc, colocs)
    mse_perf.add_result(iie.constants.GNN, avail, 'Co-detected', fraction)
    mse_perf.add_result(iie.constants.GNN, trans, 'Transitivity', 1-fraction)

    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid_random(coloc_gnn_t, pred_mc, colocs)
    mse_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction)
    mse_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction)

    acc_perf.get_df().to_csv('/g/alexandr/tim/GNN_perf_acc.csv')
    mse_perf.get_df().to_csv('/g/alexandr/tim/GNN_perf_mse.csv')













# %%

import pandas as pd
from typing import Final, Dict

acc_perf_df = pd.concat([pd.read_csv('/g/alexandr/tim/GNN_perf_acc.csv').rename(columns={'Scenario': 'Model'}),
                         pd.read_csv('/g/alexandr/tim/CLR_perf_acc.csv')])
mse_perf_df = pd.concat([pd.read_csv('/g/alexandr/tim/GNN_perf_mse.csv').rename(columns={'Scenario': 'Model'}),
                         pd.read_csv('/g/alexandr/tim/CLR_perf_mse.csv')])


# Exclude the rows selfContrast, colocContrast, and regContrast from the dataframe
acc_perf_df = acc_perf_df[~acc_perf_df['Model'].isin(['selfContrast', 'colocContrast', 'regContrast'])]
mse_perf_df = mse_perf_df[~mse_perf_df['Model'].isin(['selfContrast', 'colocContrast', 'regContrast'])]

acc_perf_df = acc_perf_df.replace('BMC', 'BioMedCLIP')
acc_perf_df = acc_perf_df.replace('infoNCE', 'infoNCE CNN')

mse_perf_df = mse_perf_df.replace('BMC', 'BioMedCLIP')
mse_perf_df = mse_perf_df.replace('infoNCE', 'infoNCE CNN')


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

MEAN_COLOC : Final[str] = 'Mean Coloc'
UMAP : Final[str] = 'UMAP'
GNN : Final[str] = 'GNN'
RANDOM : Final[str] = 'Random'
BMC : Final[str] = 'BioMedCLIP'
INFO_NCE : Final[str] = 'infoNCE CNN'

# Colors
MODEL_PALLETE : Final[Dict[str, str]] = {
    MEAN_COLOC: 'darkgrey',
    UMAP: 'gray',
    GNN: '#1aae54ff',
    RANDOM: 'white',
    BMC: '#229cefff',
    INFO_NCE: '#22efecff'
}

plot_order = [MEAN_COLOC, GNN, UMAP, BMC, INFO_NCE, RANDOM]
# %%
# Accuracy 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Co-detected']
sns.violinplot(data=data_df, 
               x='Model', y='Accuracy', ax=ax1, palette=MODEL_PALLETE,
               order=[x for x in plot_order if x in data_df['Model'].unique()],
               cut=0)
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
ax1.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ticks = ax1.get_yticklabels()
ticks[-1].set_weight('bold')

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()
sns.violinplot(data=data_df, 
               x='Model', y='Accuracy', ax=ax2, palette=MODEL_PALLETE, 
               order=[x for x in plot_order if x in data_df['Model'].unique()],
               cut=0)
frac = data_df['Fraction'].mean()
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', va='top')
ticks = ax2.get_yticklabels()
ticks[-1].set_weight('bold')

plt.tight_layout()

plt.savefig('/g/alexandr/tim/tmp/acc_ion_fig.pdf', bbox_inches='tight')

# %%
# MSE

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = mse_perf_df[mse_perf_df['Evaluation']=='Co-detected']
sns.violinplot(data=data_df, 
               x='Model', y='MSE', ax=ax1, palette=MODEL_PALLETE,
               order=[x for x in plot_order if x in data_df['Model'].unique()],
               cut=0)
ax1.set_ylabel(f'MSE (Co-detected)')
sns.despine(offset=5, trim=False, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', va='top')
ticks = ax1.get_yticklabels()
ax1.set_ylim(0, .5)
ticks[0].set_weight('bold')

data_df = mse_perf_df[mse_perf_df['Evaluation']=='Transitivity'].dropna()
sns.violinplot(data=data_df, 
               x='Model', y='MSE', ax=ax2, palette=MODEL_PALLETE,
               order=[x for x in plot_order if x in data_df['Model'].unique()],
               cut=0)
frac = data_df['Fraction'].mean()
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(frac))
ax2.set_ylabel(f'MSE (Transitivity)')
sns.despine(offset=5, trim=False, ax=ax2)
ax2.set_ylim(0, .5)
# Rotate x-axis labels by 45 degrees
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', va='top')
ticks = ax2.get_yticklabels()
ticks[0].set_weight('bold')
ax2.set_yticklabels(ticks)
plt.tight_layout()

plt.savefig('/g/alexandr/tim/tmp/mse_ion_fig.pdf', bbox_inches='tight')

# %%
