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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import ionimage_embedding as iie


# %%
import os
os.system('nvidia-smi')




# %%
iidata = iie.dataloader.IonImage_data.IonImagedata_transitivity(iie.datasets.KIDNEY_SMALL, 
                        test=0.01, val=0.1, 
                        cache=True, cache_folder='/scratch/model_testing',
                        colocml_preprocessing=True, 
                        fdr=.1, batch_size=40, knn=False,
                        transformations=None, # T.RandomRotation(degrees=(0, 360)) 
                        maxzero=.9, min_codetection=2)

colocs = iie.dataloader.get_coloc_model.get_coloc_model(iidata)

num_nan = np.isnan(colocs.test_mean_coloc).sum().sum()
frac_nan =  num_nan / (colocs.test_mean_coloc.shape[0] * colocs.test_mean_coloc.shape[1])

print('Number of NaNs:', num_nan)
print('Fraction of NaNs:', frac_nan)
print('Train size: ', [x.shape[0] for x in colocs.train_coloc.values()])
print('Test size: ', [x.shape[0] for x in colocs.test_coloc.values()])
print('Val size: ', [x.shape[0] for x in colocs.val_coloc.values()])



# %%
nums = []
fracs = []
tt = []

N_REPS = 5

for i in np.linspace(0.01, 0.1, 10):
    for rep in range(N_REPS):
        iidata = iie.dataloader.IonImage_data.IonImagedata_transitivity(iie.datasets.KIDNEY_SMALL, 
                                                                        test=i, val=0.1, 
                        cache=True, cache_folder='/scratch/model_testing',
                        colocml_preprocessing=True, 
                        fdr=.1, batch_size=40, knn=False,
                        transformations=None, # T.RandomRotation(degrees=(0, 360)) 
                        maxzero=.9, min_codetection=2)

        colocs = iie.dataloader.get_coloc_model.get_coloc_model(iidata)

        num_nan = np.isnan(colocs.test_mean_coloc).sum().sum()
        frac_nan =  num_nan / (colocs.test_mean_coloc.shape[0] * colocs.test_mean_coloc.shape[1])

        nums.append(num_nan)
        fracs.append(frac_nan)
        tt.append(round(i, 4))

df = pd.DataFrame({'nan': nums, 'frac': fracs, 'test_fraction': tt})

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.boxplot(data=df, x='test_fraction', y='nan', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
sns.boxplot(data=df, x='test_fraction', y='frac', ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
