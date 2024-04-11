# %%
import sys
import numpy as np

import ionimage_embedding as iie


# Hyperparameters
# #####################
# Hyperparameters
# #####################

# %%
# test
test = 1
# val
val = 1
# accuracy top-k
top_acc = 3
# Dataset
DSID = iie.datasets.KIDNEY_LARGE
DS_NAME = 'KIDNEY_LARGE'



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

encod_l = ['onehot', 'learned', 'atom_composition']


hyperparams = hyperparams_transitivity
RANDOM_NETWORK = False

# min_images
min_images = hyperparams['top_k'] + hyperparams['bottom_k'] + 1

acc_perf = iie.logger.PerformanceLogger('Model','Accuracy', 'Evaluation', 'Fraction', 
                                        '#NaN', 'fraction NaN', 'Test fraction')
mse_perf = iie.logger.PerformanceLogger('Model', 'MSE',     'Evaluation', 'Fraction', 
                                        '#NaN', 'fraction NaN', 'Test fraction')

# %%
# RUN ID
RUN_ID = str(sys.argv[1])

acc_file = f'/g/alexandr/tim/ACC_trans_GNN_encod_{RUN_ID}.csv'
mse_file = f'/g/alexandr/tim/MSE_trans_GNN_encod_{RUN_ID}.csv'

for count, test_value in enumerate(np.linspace(0.001, 0.1, 50)):
    print('# #######')
    print(f'# Iteration {count}')
    print('# #######')

    try:
        for encod in encod_l:
            # II Data
            iidata = iie.dataloader.IonImage_data.IonImagedata_transitivity(
                DSID, test=test_value, val=.1, transformations=None, fdr=.1,
                min_images=min_images, maxzero=.9, batch_size=10, knn=False,
                colocml_preprocessing=True, cache=True)

            iidata.sample_sets()

            colocs = iie.dataloader.get_coloc_model.get_coloc_model(iidata, device='cuda')

            # Transitivity info
            num_nan = np.isnan(colocs.test_mean_coloc).sum().sum()
            frac_nan =  num_nan / (colocs.test_mean_coloc.shape[0] * colocs.test_mean_coloc.shape[1])

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
            hidden_factor=.5
            if encod == 'atom_composition':
                hidden_factor = 2

            model = iie.models.gnn.gnnd.gnnDiscrete(data=dat, latent_dims=hyperparams['latent_size'], 
                                    encoding = encod, embedding_dims=40,
                                    lr=hyperparams['lr'], training_epochs=130, 
                                    early_stopping_patience=hyperparams['early_stopping_patience'],
                                    lightning_device='gpu', loss=hyperparams['loss_type'],
                                    activation=hyperparams['activation'], num_layers=hyperparams['num_layers'],
                                    gnn_layer_type=hyperparams['gnn_layer_type'], 
                                    hidden_factor=hidden_factor)

            mylogger = model.train()

            pred_mc = colocs.test_mean_coloc

            coloc_embedding = iie.evaluation.latent.coloc_umap_iid(colocs, k=3, n_components=5)
            coloc_cu = iie.evaluation.latent.latent_colocinference(
                coloc_embedding, 
                colocs.test_dataset.ion_labels)
            
            pred_gnn_t = iie.evaluation.latent.latent_gnn(model, dat, graph='training')
            coloc_gnn_t = iie.evaluation.latent.latent_colocinference(pred_gnn_t, colocs.test_dataset.ion_labels)
            
            
            # Accuracy
            

            avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=coloc_cu,
                                                                            agg_coloc_pred=pred_mc,
                                                                            colocs=colocs, 
                                                                            top=top_acc)
            acc_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            acc_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)
            
            avail, trans, _ = iie.evaluation.metrics.coloc_top_acc_iid(latent=pred_mc,
                                                                    agg_coloc_pred=pred_mc,
                                                                    colocs=colocs, 
                                                                    top=top_acc)
            acc_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 
                                fraction, # We use the fraction as computeed by methods with a latent space
                                num_nan, frac_nan, test_value)

            avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=coloc_gnn_t, 
                                                                            agg_coloc_pred=pred_mc,
                                                                            colocs=colocs, 
                                                                            top=top_acc)
            acc_perf.add_result(iie.constants.GNN+'_'+encod, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            acc_perf.add_result(iie.constants.GNN+'_'+encod, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)

            avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid_random(pred_mc, 
                                                                                    agg_coloc_pred=pred_mc,
                                                                                    colocs=colocs, 
                                                                                    top=top_acc) 
            acc_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            acc_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)
            
            # MSE
            avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(pred_mc, pred_mc, colocs)
            mse_perf.add_result(iie.constants.MEAN_COLOC, avail, 'Co-detected', 1, 
                                num_nan, frac_nan, test_value)

            avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(coloc_cu, pred_mc, colocs)
            mse_perf.add_result(iie.constants.UMAP, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            mse_perf.add_result(iie.constants.UMAP, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)

            avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(coloc_gnn_t, pred_mc, colocs)
            mse_perf.add_result(iie.constants.GNN+'_'+encod, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            mse_perf.add_result(iie.constants.GNN+'_'+encod, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)

            avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid_random(coloc_gnn_t, pred_mc, colocs)
            mse_perf.add_result(iie.constants.RANDOM, avail, 'Co-detected', fraction, 
                                num_nan, frac_nan, test_value)
            mse_perf.add_result(iie.constants.RANDOM, trans, 'Transitivity', 1-fraction, 
                                num_nan, frac_nan, test_value)

            acc_perf.get_df().to_csv(acc_file)
            mse_perf.get_df().to_csv(mse_file)

    except ValueError:
        print('#################################')
        print('Data loading error at: ', test_value)
        print('#################################')

