# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T
import seaborn as sns


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

import ionimage_embedding as iie

# %%

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')


# %%
# prerequisites

Ntimes = 30
filename_acc = '/g/alexandr/tim/CLR_perf_acc.csv'
filename_mse = '/g/alexandr/tim/CLR_perf_mse.csv'
top_acc = 3
coloc_agg='mean'
device='cuda'

DS_NAME = 'KIDNEY_SMALL'
dataset = iie.datasets.KIDNEY_SMALL

acc_perf = iie.logger.PerformanceLogger(scenario='Model',metric='Accuracy', 
                                        evaluation='Evaluation', fraction='Fraction')

mse_perf = iie.logger.PerformanceLogger(scenario='Model',metric='MSE',
                                        evaluation='Evaluation', fraction='Fraction')
# %% InfoNCE
for i in range(Ntimes):
    lt = 'infoNCE'
    print(f'{lt} {i}/{Ntimes}')

    iid = iie.dataloader.IonImage_data.IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = iie.models.crl.crl.CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=50, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=False, knn=True, lr=0.08)
    
    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iid)
    pred_mc = colocs.test_mean_coloc

    mylogger = model.train(logger=False)

    latent_model = iie.evaluation.latent.latent_iid(model, origin='train')
    model_coloc_inferred = iie.evaluation.latent.latent_colocinference(latent_model, 
                                                                       colocs.test_dataset.ion_labels)
    
    # Acc
    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=model_coloc_inferred, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(lt, avail, 'Co-detected', fraction)
    acc_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(model_coloc_inferred, 
                                                                  pred_mc, colocs)
    mse_perf.add_result(lt, avail, 'Co-detected', fraction)
    mse_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

acc_perf.get_df().to_csv(filename_acc)
mse_perf.get_df().to_csv(filename_mse)


# %% selfContrast
for i in range(Ntimes):
    lt = 'selfContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = iie.dataloader.IonImage_data.IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = iie.models.crl.crl.CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iid)
    pred_mc = colocs.test_mean_coloc

    mylogger = model.train(logger=True)

    latent_model = iie.evaluation.latent.latent_iid(model, origin='train')
    model_coloc_inferred = iie.evaluation.latent.latent_colocinference(latent_model, 
                                                                       colocs.test_dataset.ion_labels)
    
    # Acc
    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=model_coloc_inferred, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(lt, avail, 'Co-detected', fraction)
    acc_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(model_coloc_inferred, 
                                                                  pred_mc, colocs)
    mse_perf.add_result(lt, avail, 'Co-detected', fraction)
    mse_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

acc_perf.get_df().to_csv(filename_acc)
mse_perf.get_df().to_csv(filename_mse)


# %% colocContrast
for i in range(Ntimes):
    lt = 'colocContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = iie.dataloader.IonImage_data.IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = iie.models.crl.crl.CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iid)
    pred_mc = colocs.test_mean_coloc

    mylogger = model.train(logger=True)

    latent_model = iie.evaluation.latent.latent_iid(model, origin='train')
    model_coloc_inferred = iie.evaluation.latent.latent_colocinference(latent_model, 
                                                                       colocs.test_dataset.ion_labels)
    
    # Acc
    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=model_coloc_inferred, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(lt, avail, 'Co-detected', fraction)
    acc_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(model_coloc_inferred, 
                                                                  pred_mc, colocs)
    mse_perf.add_result(lt, avail, 'Co-detected', fraction)
    mse_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

acc_perf.get_df().to_csv(filename_acc)
mse_perf.get_df().to_csv(filename_mse)

# %% regContrast
for i in range(Ntimes):
    lt = 'regContrast'
    print(f'{lt}: {i}/{Ntimes}')

    iid = iie.dataloader.IonImage_data.IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    model = iie.models.crl.crl.CRL(iid, num_cluster=20, lightning_device='gpu', activation='relu', loss_type=lt,
                cae=False, training_epochs=30, architecture='resnet18', resnet_pretrained=True,
                initial_upper=90, initial_lower=22, upper_iteration=0.0, lower_iteration=0.0, 
                dataset_specific_percentiles=True, knn=True, lr=0.08)
    
    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iid)
    pred_mc = colocs.test_mean_coloc

    mylogger = model.train(logger=True)

    latent_model = iie.evaluation.latent.latent_iid(model, origin='train')
    model_coloc_inferred = iie.evaluation.latent.latent_colocinference(latent_model, 
                                                                       colocs.test_dataset.ion_labels)
    
    # Acc
    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=model_coloc_inferred, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(lt, avail, 'Co-detected', fraction)
    acc_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(model_coloc_inferred, 
                                                                  pred_mc, colocs)
    mse_perf.add_result(lt, avail, 'Co-detected', fraction)
    mse_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

acc_perf.get_df().to_csv(filename_acc)
mse_perf.get_df().to_csv(filename_mse)

# %% BMC
for i in range(Ntimes):
    lt = 'BMC'
    print(f'{lt}: {i}/{Ntimes}')

    iid = iie.dataloader.IonImage_data.IonImagedata_random(dataset, test=0.3, val=0.1, 
                    cache=True, cache_folder='/scratch/model_testing',
                    colocml_preprocessing=True, fdr=.1, batch_size=60, 
                    transformations=T.RandomRotation((0, 360)), # T.RandomRotation(degrees=(0, 360)) 
                    maxzero=.9, vitb16_compatible=False)
    
    
    colocs = iie.dataloader.get_coloc_model.get_coloc_model(iid)
    pred_mc = colocs.test_mean_coloc

    bmc = iie.models.biomedclip.BioMedCLIP(data=iid)

    latent_model = iie.evaluation.latent.latent_iid(bmc, origin='train')
    model_coloc_inferred = iie.evaluation.latent.latent_colocinference(latent_model, 
                                                                       colocs.test_dataset.ion_labels)
    
    # Acc
    avail, trans, fraction = iie.evaluation.metrics.coloc_top_acc_iid(latent=model_coloc_inferred, 
                                                                      agg_coloc_pred=pred_mc,
                                                                      colocs=colocs, 
                                                                      top=top_acc)
    acc_perf.add_result(lt, avail, 'Co-detected', fraction)
    acc_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

    # MSE
    avail, trans, fraction = iie.evaluation.metrics.coloc_mse_iid(model_coloc_inferred, 
                                                                  pred_mc, colocs)
    mse_perf.add_result(lt, avail, 'Co-detected', fraction)
    mse_perf.add_result(lt, trans, 'Transitivity', 1-fraction)

acc_perf.get_df().to_csv(filename_acc)
mse_perf.get_df().to_csv(filename_mse)




# %%
# Accuracy 
df = acc_perf.get_df()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Co-detected'], x='Model', y='Accuracy', ax=ax1, 
               cut=0)
ax1.set_ylabel(f'Top-{top_acc} Accuracy (Co-detected)')
ax1.set_ylim(0, 1)

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Model', y='Accuracy', ax=ax2,
               cut=0)
frac = df[df['Evaluation']=='Transitivity']['Fraction'].mean()
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(frac))
ax2.set_ylabel(f'Top-{top_acc} Accuracy (Transitivity)')
ax2.set_ylim(0, 1)

fig.suptitle(f'{DS_NAME}, Leave out datasets')
plt.show()

# %%
# MSE
df = mse_perf.get_df()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=df[df['Evaluation']=='Co-detected'], x='Model', y='MSE', ax=ax1,
               cut=0)
ax1.set_ylabel(f'MSE (Co-detected)')

sns.violinplot(data=df[df['Evaluation']=='Transitivity'], x='Model', y='MSE', ax=ax2,
               cut=0)
frac = df[df['Evaluation']=='Transitivity']['Fraction'].mean()
ax2.set_title('Mean transitivity fraction: {:.2f}'.format(frac))
ax2.set_ylabel(f'MSE (Transitivity)')

fig.suptitle(f'{DS_NAME}, Leave out datasets')
plt.show()# %%

# %%
