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
import pickle
import os.path as osp
from typing import Literal

import ionimage_embedding as iie

# %%
import os
os.system('nvidia-smi')


# %%
from molmass import Formula
from metaspace import SMInstance
from typing import Dict
from sklearn.preprocessing import StandardScaler

sm = SMInstance()

ds = sm.dataset(id='2021-12-09_20h48m20s')
results = ds.results().reset_index()
ai = ds.all_annotation_images()

# %%
def get_composition_dict(sumformula: str) -> Dict[str, int]:
    form = Formula(sumformula)
    out = {}
    for k in form.composition().keys():
        out[k] = form.composition()[k].count

    return out

comps = np.array([get_composition_dict(x.formula+x.adduct) for x in ai])

# Create mapper:
def get_atom_mapper(comps):
    atoms = np.unique([k for x in comps for k in x.keys()])
    atom_mapper = {a: i for i, a in enumerate(atoms)}
    return atom_mapper

def get_atom_counts(comps, atom_mapper):
    out = np.zeros((len(comps), len(atom_mapper)))
    for i, comp in enumerate(comps):
        for k, v in comp.items():
            out[i, atom_mapper[k]] = v
    return out

atom_mapper = get_atom_mapper(comps)

atom_counts = get_atom_counts(comps, atom_mapper)

scaled_counts = StandardScaler().fit_transform(atom_counts)
print(scaled_counts)

# %%
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
top_acc = 3
# Dataset
DSID = iie.datasets.KIDNEY_LARGE
# Number of bootstraps
N_BOOTSTRAPS = 10

latent_size = 27
top_k = 2
bottom_k = 2
encoding: Literal['onehot', 'learned', 'atom_composition'] = 'atom_composition'
early_stopping_patience = 2
gnn_layer_type = 'GATv2Conv'
loss_type = 'coloc'
num_layers = 1
lr = 0.005945


# %%
dat = iie.dataloader.ColocNet_data.ColocNetData_discrete(DSID, test=test, val=val, 
                    cache_images=True, cache_folder=iie.constants.CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=top_k, bottom_k=bottom_k, random_network=False
                    )


# %%
# ###########
# Train model
# ###########
trans_l = []
avail_l = []

for i in range(N_BOOTSTRAPS):
    print(f'# Iteration {i}')
    dat.sample_sets()

    model = iie.models.gnn.gnnd.gnnDiscrete(data=dat, 
                        latent_dims=latent_size, 
                        encoding = encoding, # type: ignore
                        embedding_dims=40,
                        lr=lr, 
                        training_epochs=130, 
                        early_stopping_patience=early_stopping_patience,
                        lightning_device='gpu', 
                        loss=loss_type, # type: ignore
                        activation='none', 
                        num_layers=num_layers,
                        gnn_layer_type=gnn_layer_type,
                        hidden_factor=3) # type: ignore


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

    avail, trans, _ = iie.evaluation.metrics.coloc_top_acc_gnn(coloc_gnn_t, 
                                                               dat, pred_mc, top=top_acc)
    
    trans_l.append(trans)
    avail_l.append(avail)

res = - (np.mean(trans_l)*2 + np.mean(avail_l))

print(res)

# %%
