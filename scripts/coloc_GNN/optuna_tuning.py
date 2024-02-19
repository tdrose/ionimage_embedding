import numpy as np
import pickle
import os.path as osp

import optuna
from optuna.samplers import TPESampler

import ionimage_embedding as iie

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
N_BOOTSTRAPS = 30




def objective(trial: optuna.Trial):

    # #########################
    # Optimized Hyperparameters
    # #########################
    # Latent size
    latent_size = trial.suggest_int('Latent size', 3, 40)
    # Top-k
    top_k = trial.suggest_int('Top-k', 2, 15)
    # Bottom-k
    bottom_k = trial.suggest_int('Bottom-k', 2, 15)
    # encoding
    encoding = trial.suggest_categorical('Encoding', ['onehot', 'learned'])
    # early_stopping_patience
    early_stopping_patience = trial.suggest_int('Early stopping patience', 1, 10)
    # GNN layer type
    gnn_layer_type = trial.suggest_categorical('GNN layer type', ['GCNConv', 'GATv2Conv'])
    # Loss type
    loss_type = trial.suggest_categorical('Loss type', ['recon', 'coloc'])
    # Number of layers
    num_layers = trial.suggest_int('Number of layers', 1, 3)
    # learning rate
    lr = trial.suggest_float('learning rate', 0.0001, 0.01)


    # #########
    # Load data
    # #########
    dat = iie.dataloader.ColocNet_data.ColocNetData_discrete(DSID, test=test, val=val, 
                    cache_images=True, cache_folder=iie.constants.CACHE_FOLDER,
                    colocml_preprocessing=True, 
                    fdr=.1, batch_size=1, min_images=min_images, maxzero=.9,
                    top_k=top_k, bottom_k=bottom_k, random_network=False
                    )
    
    # ###########
    # Train model
    # ###########
    trans_l = []
    avail_l = []

    for i in range(N_BOOTSTRAPS):
    
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
                            gnn_layer_type=gnn_layer_type) # type: ignore


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

        avail, trans, _ = iie.evaluation.metrics.coloc_top_acc_gnn(
            coloc_gnn_t, dat, pred_mc, top=top_acc)
        
        trans_l.append(trans)
        avail_l.append(avail)

    
    
    trans = np.nanmean(trans_l)
    avail = np.nanmean(avail_l)

    if np.isnan(trans):
        trans = 0
    if np.isnan(avail):
        avail= 0

    res = - (trans*100 + avail)

    # Create a dictionary of all hyperparameters
    hyperparameters = {
        'Latent size': latent_size,
        'Top-k': top_k,
        'Bottom-k': bottom_k,
        'Encoding': encoding,
        'Early stopping patience': early_stopping_patience,
        'GNN layer type': gnn_layer_type,
        'Loss type': loss_type,
        'Number of layers': num_layers,
        'learning rate': lr
    }

    print('++++++++++++++++++++++++++++')
    print('+ Current trial result {}'.format(res))
    print('+ Current hyperparameters:')
    print('+ ', hyperparameters)
    print('++++++++++++++++++++++++++++')
    
    return float(res) 


# Optimize study
study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, 
               n_trials=2000, 
               timeout=59*60*60, 
               catch=(ValueError, ZeroDivisionError, 
                      RuntimeError), 
               show_progress_bar=True)

print('###############')
print('Best parameters')
print(study.best_params)
print('###############')

pickle.dump(study, open(osp.join(iie.constants.CACHE_FOLDER, 'GNN_tuning_transitivity2.pkl'), 'wb'))

