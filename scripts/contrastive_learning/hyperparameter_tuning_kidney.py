import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

import torch

from ionimage_embedding.models import CRL
from ionimage_embedding.models import ColocModel
from ionimage_embedding.dataloader.IonImage_data import IonImagedata_random
from ionimage_embedding.models.coloc.utils import torch_cosine
from ionimage_embedding.evaluation import evaluation_quantile_overlap, ds_coloc_convert
from ionimage_embedding.evaluation.crl_inference import crl_ds_coloc, crl_fulllatent_coloc, crl_latent_coloc, crl_latent_inference

import optuna
from optuna.samplers import TPESampler


ds_list = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s',
    '2022-12-07_01h02m53s',
    '2022-12-07_01h01m06s',
    '2022-11-28_22h24m25s',
    '2022-11-28_22h23m30s'
                  ]


device='cuda'

quant = .25



def objective(trial: optuna.Trial):
    
    # ###
    # Params
    # ###
    num_cluster = trial.suggest_int('Number cluster', 5, 40)
    batchsize = trial.suggest_int('Batch size', 50, 130)
    # iu = trial.suggest_int('Initial upper', 70, 95)
    # il = trial.suggest_int('Initial lower', 5, 40)
    # us = trial.suggest_float('Upper step', 0., .8)
    # ls = trial.suggest_float('Lower step', 0., .8)
    lr = trial.suggest_float('learning rate', 0.001, 0.1)
    # dsp = trial.suggest_categorical('DS percentiles', [True, False])
    # knn = trial.suggest_categorical('KNN', [True, False])
    # epochs = trial.suggest_int('Epochs', 10, 30)
    activation = trial.suggest_categorical('Activation', ['softmax', 'relu', 'sigmoid'])



    # Load data
    crldat = IonImagedata_random(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=batchsize, transformations=None)

    model = CRL(crldat,
            num_cluster=num_cluster,
            initial_upper=90,
            initial_lower=10,
            upper_iteration=.1,
            lower_iteration=.1,
            dataset_specific_percentiles=True,
            knn=True, 
            lr=lr,
            pretraining_epochs=10,
            training_epochs=20, # epochs,
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            activation=activation,
            clip_gradients=None
            )

    # Training
    model.train(logger=False)

    # Inferrence
    cdsc = crl_ds_coloc(model, model.data.train_dataset, device=device)
    crl_test_colocs, _ = crl_latent_inference(cdsc, model.data.test_dataset.ion_labels, agg='mean')

    colocs = ColocModel(crldat)
    colocgt = colocs.quantile_gt(test_il = colocs.data.test_dataset.ion_labels, 
                                    test_dsl=colocs.data.test_dataset.dataset_labels,
                                    full_colocs=colocs.full_coloc, 
                                    upper_quantile=1.-quant, lower_quantile=quant)

    crltesteval = colocs.quantile_eval(test_il = colocs.data.test_dataset.ion_labels, 
                                    test_dsl=colocs.data.test_dataset.dataset_labels,
                                    test_colocs=crl_test_colocs,
                                    upper_quantile=1.-quant, lower_quantile=quant)
    
    evaluation_dict = {'ground_truth': colocgt, 'predictions': {'crl': crltesteval}, 'uq': 1-quant, 'lq': quant}

    df = evaluation_quantile_overlap(evaluation_dict)

    # Negative, since we are minimizing
    return - df[df['model']=='crl']['accuracy'].mean()


# Optimize study
study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, n_trials=200, timeout=None, 
               catch=(ValueError, ZeroDivisionError, ), 
               show_progress_bar=True)

print('###############')
print('Best parameters')
print(study.best_params)
print('###############')