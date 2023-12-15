# %%
import matplotlib.pyplot as plt
import torchvision.transforms as T

from ionimage_embedding.models import CRL, ColocModel
from ionimage_embedding.dataloader import CRLdata
from ionimage_embedding.evaluation.scoring import (
    closest_accuracy_aggcoloc,
    closest_accuracy_latent,
    closest_accuracy_random,
    compute_ds_coloc,
    latent_dataset_silhouette
)


# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
except ImportError:
    # Not in IPython, continue with normal Python code
    pass


# %%

# Check if session connected to the correct GPU server
import os
os.system('nvidia-smi')


# %%
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

crldat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=100, 
                 transformations=T.RandomRotation(degrees=(0, 360)) , # T.RandomRotation(degrees=(0, 360)) 
                 maxzero=.9)

# print(np.isnan(crldat.full_dataset.images).any())
# plt.imshow(crldat.full_dataset.images[10])
# plt.show()



# %%
colocs = ColocModel(crldat)



# %%
model = CRL(crldat,
            num_cluster=50,
            initial_upper=90, # 93
            initial_lower=22, # 37
            upper_iteration=0.13, # .8
            lower_iteration=0.24, # .8
            dataset_specific_percentiles=False, # True
            knn=True, # False
            lr=0.18,
            pretraining_epochs=10,
            training_epochs=40, # 30
            cae_encoder_dim=2,
            lightning_device='gpu',
            cae=False,
            cnn_dropout=0.01,
            activation='relu', # softmax
            loss_type='colocContrast', # 'selfContrast', 'colocContrast', 'regContrast',
            resnet='resnet18',
            resnet_pretrained=False,
            clip_gradients=None
            )

# %%
device='cuda'
mylogger = model.train(logger=True)

# %%
plt.plot(mylogger.logged_metrics['Validation loss'], label='Validation loss', color='orange')
plt.plot(mylogger.logged_metrics['Training loss'], label='Training loss', color='blue')
plt.legend()
plt.show()


# %%
ds = 'test'
top = 5
coloc_agg='mean'
dsc_dict = compute_ds_coloc(model, dataset=ds)


print('Model accuracy: ', closest_accuracy_latent(dsc_dict, colocs, top=top, dataset=ds))
print('Random accuracy: ', closest_accuracy_random(dsc_dict, colocs, top=top, dataset=ds))
if ds == 'test':
    print(f'{coloc_agg} accuracy: ', closest_accuracy_aggcoloc(colocs, top=top))

print('Silhouette: ', latent_dataset_silhouette(model, dataset=ds))

# %%
