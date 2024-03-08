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


import numpy as np

import ionimage_embedding as iie


# %%
import os
os.system('nvidia-smi')


# %%
# #####################
# Fixed Hyperparameters
# #####################
# device
# min_images
min_images = 30
# test
test = 5
# val
val = 3
# accuracy top-k
top_acc = 3
# Dataset
DSID = iie.datasets.KIDNEY_LARGE
# Number of bootstraps
N_BOOTSTRAPS = 100

# Random network
RANDOM_NETWORK = False


# Latent size
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


# %%
iidata = iie.dataloader.IonImage_data.IonImagedata_random(DSID, test=.1, val=.1, transformations=None, fdr=.1,
                             min_images=min_images, maxzero=.9, batch_size=10, knn=False,
                             colocml_preprocessing=True, cache=True)


colocs = iie.dataloader.get_coloc_model.get_coloc_model(iidata)

# %%
dat = iie.dataloader.ColocNet_data.ColocNetData_discrete(DSID, test=test, val=val, 
                    cache_images=True, cache_folder=iie.constants.CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=hyperparams['top_k'], bottom_k=hyperparams['bottom_k'], 
                    random_network=RANDOM_NETWORK,
                    use_precomputed=True,
                    ds_labels=iidata.full_dataset.dataset_labels,
                    ion_labels=iidata.full_dataset.ion_labels,
                    coloc=colocs.full_coloc,
                    dsl_int_mapper=iidata.dsl_int_mapper,
                    ion_int_mapper=iidata.ion_int_mapper,
                    n_ions=len(iidata.full_dataset.ion_labels.unique())
                    )


# %%
# ###########
# Train model
# ###########

dat.sample_sets()

model = iie.models.gnn.gnnd.gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                        encoding = hyperparams['encoding'], embedding_dims=40,
                        lr=hyperparams['lr'], training_epochs=130, 
                        early_stopping_patience=hyperparams['early_stopping_patience'],
                        lightning_device='gpu', loss=hyperparams['loss_type'],
                        activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                        gnn_layer_type=hyperparams['gnn_layer_type'])


_ = model.train()

# ##########
# Evaluation
# ##########
pred_mc, _ = iie.evaluation.utils_gnn.mean_coloc_test(dat)

pred_gnn_t = iie.evaluation.latent.latent_gnn(model, dat, graph='training')
coloc_gnn_t = iie.evaluation.latent.latent_colocinference(
    pred_gnn_t, 
    iie.evaluation.utils_gnn.coloc_ion_labels(dat, dat._test_set)
    )

avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_gnn(coloc_gnn_t, 
                                                                  dat, pred_mc, top=top_acc)
print('Available: ', avail)
print('Transitivity: ', trans)
print('Fraction: ', fraction)



# %% Plotting parameter
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=XBIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({'text.color': "595a5cff",
                     'axes.labelcolor': "595a5cff",
                     'xtick.color': "595a5cff",
                     'ytick.color': "595a5cff",})

def draw_umapaxis(ax, length_mul=.1, yoffset=0.):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    deltax = xlims[1] - xlims[0]
    deltay = ylims[1] - ylims[0]
    
    # X-Annotation
    ax.annotate('', xy=(xlims[0]+length_mul*deltax, ylims[0]+yoffset), xytext=(xlims[0], ylims[0]+yoffset), 
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('UMAP1', (xlims[0]+length_mul*deltax, ylims[0]+yoffset), ha='left', va='center', fontsize=XSMALLER_SIZE)
    
    ax.annotate('', xy=(xlims[0], ylims[0]+length_mul*deltay+yoffset), xytext=(xlims[0], ylims[0]+yoffset), 
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('UMAP2', (xlims[0], ylims[0]+length_mul*deltay+yoffset), ha='center', va='bottom', fontsize=XSMALLER_SIZE)

# %%
# ####################
# Latent Visualization
# ####################


import scanpy as sc
import pandas as pd
# %%

pd.Series(dat.ion_int_mapper)
# %%
adata = sc.AnnData(X=pred_gnn_t, 
                   obs=pd.DataFrame({'ion_label': pd.Series(dat.ion_int_mapper)[pred_gnn_t.index]})
                   )

adata.obs[['molecule', 'ion_label']] = adata.obs['ion_label'].str.split('+', expand=True)
adata.obs[['c', 'rest']] = adata.obs['molecule'].str.split('H', expand=True)

# %%
hmdb_v4 = pd.read_csv('/g/alexandr/tim/metaspace_evaluation/databases/HMDB_v4.csv')
hmdb_v4 = hmdb_v4[['name', 'chemical_formula', 'super_class', 'sub_class', 'class']]

metadata_dict = {'class': [], 'sub_class': [], 'super_class': [], 'names': [], 'ion_label': []}

for i in range(adata.obs.shape[0]):
    mol = adata.obs.iloc[i]['molecule']
    ion = adata.obs.index.values[i]
    tmp = hmdb_v4[hmdb_v4['chemical_formula']==mol]

    if tmp.shape[0] == 0:
        metadata_dict['class'].append('unknown')
        metadata_dict['sub_class'].append('unknown')
        metadata_dict['super_class'].append('unknown')
        metadata_dict['names'].append('unknown')
        metadata_dict['ion_label'].append(ion)
        continue
    
    metadata_dict['class'].append(tmp['class'].values[0])
    metadata_dict['sub_class'].append(tmp['sub_class'].values[0])
    metadata_dict['super_class'].append(tmp['super_class'].values[0])
    
    metadata_dict['names'].append(list(tmp['name']))
    
    metadata_dict['ion_label'].append(ion)


# %%
adata.obs = adata.obs.join(pd.DataFrame(metadata_dict).set_index('ion_label'))

# %%
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# %%
sc.pl.pca(adata, color='sub_class', dimensions=[0, 1], size=80.)


# %%
sc.pl.umap(adata, color='class', size=80.)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

pos_umap = adata.obs.reset_index().join(
    pd.DataFrame(adata.obsm['X_umap']).rename(columns={0: 'x', 1: 'y'})).rename(columns={'leiden': 'Leiden cluster'})

sns.scatterplot(data=pos_umap, x='x', y='y', hue='Leiden cluster', s=25, ax=ax)

ax.axis('off')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Leiden cluster')

draw_umapaxis(ax, length_mul=.1, yoffset=0.1)
plt.savefig('/g/alexandr/tim/tmp/latent_umap.pdf', bbox_inches='tight')



# %%
for LEIDEN_CLUSTER in range(len(adata.obs['leiden'].unique())):
    N_MOLS = 5
    N_SAMPLES = 5

    ion_labels = iidata.full_dataset.ion_labels

    tmp = adata.obs[adata.obs['leiden']==str(LEIDEN_CLUSTER)].reset_index()

    # Sample ions
    sample_ions = np.random.choice(tmp['ion'].values.astype(int), N_MOLS, replace=False)

    # Create figure grid
    fig, axs = plt.subplots(N_MOLS, N_SAMPLES, figsize=(5, 5))

    for ax in axs.flatten():
        ax.axis('off')

    for i, ion in enumerate(sample_ions):
        # Mask the ions
        mask = iidata.full_dataset.ion_labels == ion

        image_idx = np.arange(len(ion_labels))[mask]

        # Shuffle image_idx
        np.random.shuffle(image_idx)

        image_idx = image_idx[:N_SAMPLES]

        for j, idx in enumerate(image_idx):
            axs[i, j].imshow(iidata.full_dataset.images[idx])

    plt.savefig(f'/g/alexandr/tim/tmp/latent_cluster_vis_{LEIDEN_CLUSTER}.pdf', bbox_inches='tight')




# %%
# ###############
# Latent variance
# ###############

import pandas as pd
from typing import Literal
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def predict_centroid_var(model: iie.models.gnn.gnnd.gnnDiscrete, 
                         data: iie.dataloader.ColocNet_data.ColocNetData_discrete,
                         distance: Literal['cosine', 'euclidean'] = 'cosine') -> pd.Series:

    # get training data
    training_data = data.dataset.index_select(data._train_set)

    # Get all predictions
    pred_dict = model.predict_multiple(training_data) # type: ignore

    # Get set of ion labels from all data objects
    ion_labels = []
    for i in range(len(training_data)):
        ion_labels.extend(list(training_data[i].x.detach().cpu().numpy())) # type: ignore
    ion_labels = list(set(ion_labels))

    # Get the variance for each prediction
    var_dict = {}
    for i in ion_labels:
        tmp = []
        for dsid, pred in pred_dict.items():
            if i in training_data[dsid].x: # type: ignore
                tmp.append(pred[training_data[dsid].x == i].detach().cpu().numpy()[0]) # type: ignore
        if len(tmp) > 1:
            a = np.stack(tmp)
            if distance == 'cosine':
                sim = cosine_similarity(a)
                mask = np.ones(sim.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                var_dict[i] = np.var(sim[mask])
            elif distance == 'euclidean':
                sim = euclidean_distances(a)
                mask = np.ones(sim.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                var_dict[i] = np.var(sim[mask])
        elif len(tmp) == 1:
            var_dict[i] = 0

    return pd.Series(var_dict)

# %%
latent_var = predict_centroid_var(model, dat, distance='euclidean')

mean_var, _ = iie.evaluation.utils_gnn.mean_coloc_train(dat, agg='var')
mean_var[mean_var==0] = np.nan

var_df = pd.DataFrame({'latent': latent_var, 'mean': mean_var.apply(np.nanmax)})
var_df=var_df.dropna()

import seaborn as sns
ax = sns.regplot(data=var_df, x='latent', y='mean', order=1, ci=None)
#ax.set_xlim(-0.0001, 0.01)

# %%
import statsmodels.api as sm

X2 = sm.add_constant(var_df[['latent']])
est = sm.OLS(var_df[['mean']], X2)
est2 = est.fit()
print(est2.summary())




# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

latent_centroid = iie.evaluation.latent.latent_gnn(model, dat, graph='training')

scaler = StandardScaler()
latent_scaled = scaler.fit_transform(latent_centroid)
N_COMP = 2
pca = PCA(n_components=N_COMP)
pca.fit(latent_scaled)

latent_pca = pca.transform(latent_scaled)

pca_df = pd.DataFrame(latent_pca, columns=[f'PC{x}' for x in range(1, N_COMP+1)], index=latent_centroid.index)

# Transform each dataset to PCA space
training_data = dat.dataset.index_select(dat._train_set)
latent_dict = model.predict_multiple(training_data) # type: ignore

pca_ds = []
for i in range(len(training_data)):
    x = training_data[i]
    dsid = int(x.ds_label[0]) # type: ignore
    v = latent_dict[i]
    index = x.x.detach().cpu().numpy() # type: ignore
    tmp = pd.DataFrame(pca.transform(scaler.transform(v.detach().cpu().numpy())), 
                       columns=[f'PC{x}' for x in range(1, N_COMP+1)], index=index)
    
    pca_ds.append(tmp)

train_df = pd.concat(pca_ds)


pca_dfl = pca_df.copy().reset_index()
pca_dfl['Cluster'] = adata.obs.reset_index()['leiden']


# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for i in pca_df.index:
    tmp = train_df.loc[i, :]
    if tmp.shape[0] > 1 and type(tmp) == pd.DataFrame:
        # select a random color
        color = np.random.rand(3,)
        for j in range(tmp.shape[0]):
            # Draw a line between the centroid and the dataset
            ax.plot([pca_df.loc[i, 'PC1'], tmp.iloc[j]['PC1']], 
                    [pca_df.loc[i, 'PC2'], tmp.iloc[j]['PC2']], 
                    color=color, alpha=0.1)

sns.scatterplot(data=pca_dfl, x='PC1', y='PC2', s=15, color='black', ax=ax)

sns.despine(offset=5, trim=False, ax=ax)
plt.savefig('/g/alexandr/tim/tmp/latent_variance.pdf', bbox_inches='tight')


# %%


sns.scatterplot(data=pca_dfl, x='PC1', y='PC2', s=15, hue='Cluster')






# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, n_init="auto").fit(pca_df)

pca_dfc = pca_df.copy()
pca_dfc['cluster'] = kmeans.labels_
sns.scatterplot(data=pca_dfc, x='PC1', y='PC2', s=15, hue='cluster')
# %%
for CLUSTER in range(len(pca_dfc['cluster'].unique())):
    N_MOLS = 5
    N_SAMPLES = 5

    ion_labels = iidata.full_dataset.ion_labels

    tmp = pca_dfc[pca_dfc['cluster']==CLUSTER].reset_index()

    # Sample ions
    if tmp.shape[0] < N_MOLS:
        sample_ions = tmp['ion'].values
    else:
        sample_ions = np.random.choice(tmp['ion'].values.astype(int), N_MOLS, replace=False)

    # Create figure grid
    fig, axs = plt.subplots(N_MOLS, N_SAMPLES, figsize=(5, 5))

    for ax in axs.flatten():
        ax.axis('off')

    for i, ion in enumerate(sample_ions):
        # Mask the ions
        mask = iidata.full_dataset.ion_labels == ion

        image_idx = np.arange(len(ion_labels))[mask]

        # Shuffle image_idx
        np.random.shuffle(image_idx)

        image_idx = image_idx[:N_SAMPLES]

        for j, idx in enumerate(image_idx):
            axs[i, j].imshow(iidata.full_dataset.images[idx])
# %%
