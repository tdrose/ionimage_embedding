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
import numpy as np
import pandas as pd
from typing import Final, Dict

import ionimage_embedding as iie




# %%
import os
os.system('nvidia-smi')

# %%
acc_file = f'ACC_trans_GNN_'
mse_file = f'MSE_trans_GNN_'



files = os.listdir('/g/alexandr/tim/')

acc_files = [f for f in files if f.startswith(acc_file)]
mse_files = [f for f in files if f.startswith(mse_file)]

acc_perf_df = pd.concat([pd.read_csv(f'/g/alexandr/tim/{f}') for f in acc_files if 'encod' not in f])
mse_perf_df = pd.concat([pd.read_csv(f'/g/alexandr/tim/{f}') for f in mse_files if 'encod' not in f])


# %%

# fraction - Output from evaluation functions about fraction of co-detected
# fraction_nan - Fraction of Actual NaNs in mean coloc matrix
# Test fraction - Inserted fraction in Transitivity dataloader

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 5))

subset = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity']

sns.scatterplot(data=subset, 
                x='Test fraction', y='fraction NaN', alpha=0.5, ax=ax1)
sns.scatterplot(data=subset, 
                x='Test fraction', y='Fraction', alpha=0.5, ax=ax2)

sns.scatterplot(data=subset, 
                x='fraction NaN', y='Fraction', alpha=0.5, ax=ax3)
plt.show()


# %%
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
    RANDOM: 'black',
    BMC: '#229cefff',
    INFO_NCE: '#22efecff'
}

plot_order = [MEAN_COLOC, GNN, UMAP, BMC, INFO_NCE, RANDOM]

# accuracy top-k
top_acc = 3


# COLUMNS: ['Unnamed: 0', 'Model', 'Accuracy', 'Evaluation', 'Fraction', '#NaN', 'fraction NaN', 'Test fraction']

# %%
# Accuracy 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'Test fraction'
y = 'Accuracy'
hue = 'Model'

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax1, hue=hue, palette=MODEL_PALLETE,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
ax1.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax1)
ticks = ax1.get_yticklabels()
ticks[-1].set_weight('bold')

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax2, hue=hue, palette=MODEL_PALLETE,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
frac = data_df['Fraction'].mean()
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax2)
ticks = ax2.get_yticklabels()
ticks[-1].set_weight('bold')

plt.tight_layout()

# plt.savefig('/g/alexandr/tim/tmp/acc_ion_fig.pdf', bbox_inches='tight')

# %%
# MSE

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'Test fraction'
y = 'MSE'
hue = 'Model'

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax1, hue=hue, palette=MODEL_PALLETE,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
ax1.set_ylabel(f'MSE (Co-detected)')
ax1.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax1)
ticks = ax1.get_yticklabels()
ticks[-1].set_weight('bold')

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()
sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax2, hue=hue, palette=MODEL_PALLETE,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )

ax2.set_ylabel(f'MSE (Transitivity)')
ax2.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax2)
ticks = ax2.get_yticklabels()
ticks[-1].set_weight('bold')

plt.tight_layout()


# %%
# %%
# Accuracy 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'fraction NaN'
y = 'Accuracy'
hue = 'Model'

sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, palette=MODEL_PALLETE, order=2,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
plt.ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
plt.ylim(0, 1)
plt.show()

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, palette=MODEL_PALLETE, order=2,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
frac = data_df['Fraction'].mean()
plt.ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
plt.ylim(0, 1)
plt.show()



# %%
# MSE 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'fraction NaN'
y = 'MSE'
hue = 'Model'

tmp = sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, palette=MODEL_PALLETE, order=2, col='Model',
               facet_kws=dict(sharex=True, sharey=False), line_kws={'color': 'black'}
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
tmp.facet_axis(0,0).set_ylabel(f'MSE (Co-detected)')
# plt.ylim(0, 1)
plt.show()

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

tmp = sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, palette=MODEL_PALLETE, order=2, col='Model',
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               facet_kws=dict(sharex=True, sharey=False), line_kws={'color': 'black'}
               )
frac = data_df['Fraction'].mean()
tmp.facet_axis(0,0).set_ylabel(f'MSE (Transitivity)')
# plt.ylim(0, 1)
plt.show()








# %%



acc_file = f'ACC_trans_GNN_'
mse_file = f'MSE_trans_GNN_'



files = os.listdir('/g/alexandr/tim/')

acc_files = [f for f in files if f.startswith(acc_file)]
mse_files = [f for f in files if f.startswith(mse_file)]

acc_perf_df = pd.concat([pd.read_csv(f'/g/alexandr/tim/{f}') for f in acc_files if 'encod' in f])
mse_perf_df = pd.concat([pd.read_csv(f'/g/alexandr/tim/{f}') for f in mse_files if 'encod' in f])
# %%
# %%
# Accuracy 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'Test fraction'
y = 'Accuracy'
hue = 'Model'

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax1, hue=hue, palette=MODEL_PALLETE,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
ax1.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax1)
ticks = ax1.get_yticklabels()
ticks[-1].set_weight('bold')

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax2, hue=hue, palette=MODEL_PALLETE,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
frac = data_df['Fraction'].mean()
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax2)
ticks = ax2.get_yticklabels()
ticks[-1].set_weight('bold')

plt.tight_layout()

# plt.savefig('/g/alexandr/tim/tmp/acc_ion_fig.pdf', bbox_inches='tight')

# %%
# MSE

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'Test fraction'
y = 'MSE'
hue = 'Model'

sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax1, hue=hue,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
ax1.set_ylabel(f'MSE (Co-detected)')
ax1.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax1)
ticks = ax1.get_yticklabels()
ticks[-1].set_weight('bold')

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()
sns.scatterplot(data=data_df, 
               x=x, y=y, ax=ax2, hue=hue,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )

ax2.set_ylabel(f'MSE (Transitivity)')
ax2.set_ylim(0, 1)
sns.despine(offset=5, trim=False, ax=ax2)
ticks = ax2.get_yticklabels()
ticks[-1].set_weight('bold')

plt.tight_layout()


# %%
# %%
# Accuracy 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'fraction NaN'
y = 'Accuracy'
hue = 'Model'

sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, order=2,
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
plt.ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
plt.ylim(0, 1)
plt.show()

data_df = acc_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, order=2,
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
frac = data_df['Fraction'].mean()
plt.ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
plt.ylim(0, 1)
plt.show()



# %%
# MSE 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Co-detected']
x = 'fraction NaN'
y = 'MSE'
hue = 'Model'

tmp = sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, order=2, col='Model',
               facet_kws=dict(sharex=True, sharey=False), line_kws={'color': 'black'}
               # order=[x for x in plot_order if x in data_df['Model'].unique()],
               )
tmp.facet_axis(0,0).set_ylabel(f'MSE (Co-detected)')
# plt.ylim(0, 1)
plt.show()

data_df = mse_perf_df[acc_perf_df['Evaluation']=='Transitivity'].dropna()

tmp = sns.lmplot(data=data_df, 
               x=x, y=y, hue=hue, order=2, col='Model',
               #order=[x for x in plot_order if x in data_df['Model'].unique()],
               facet_kws=dict(sharex=True, sharey=False), line_kws={'color': 'black'}
               )
frac = data_df['Fraction'].mean()
tmp.facet_axis(0,0).set_ylabel(f'MSE (Transitivity)')
# plt.ylim(0, 1)
plt.show()
# %%
