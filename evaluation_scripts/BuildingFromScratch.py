# Building loss from scratch for a sanity check and easier debugging

# %%
import torch
import torch.nn.functional as functional
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
# Check if connected to correct server
os.system('nvidia-smi')


# %% Playing with SimCLR
# Code from repo: https://github.com/sthalles/SimCLR/tree/master
batch_size = 6

labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

mask = torch.eye(labels.shape[0], dtype=torch.bool)
labels = labels[~mask].view(labels.shape[0], -1)

features = torch.rand((batch_size*2, 10))
features = functional.normalize(features, dim=1)
similarity_matrix = torch.matmul(features, features.T)

similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)


logits = torch.cat([positives, negatives], dim=1)

# Only true labels is generated (one hot encoding of labels also possible)
labels = torch.zeros(logits.shape[0], dtype=torch.long)

# Logits and labels are then fed to CrossEntropyLoss
# Important to note is that they only have one positive pair for each image per batch
# I did not fully understand yet how they do this in the dataloader





# %%
# Soft-Nearest Neighbors Loss
# from: https://lilianweng.github.io/posts/2021-05-31-contrastive/

# Same as above until definition of positives and negatives
batch_size = 6

labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

mask = torch.eye(labels.shape[0], dtype=torch.bool)
labels = labels[~mask].view(labels.shape[0], -1)

features = torch.rand((batch_size*2, 10))
features = functional.normalize(features, dim=1)
similarity_matrix = torch.matmul(features, features.T)

similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

# %%
temp = torch.tensor(400.)

positives = torch.exp(-similarity_matrix[labels.bool()].view(labels.shape[0], -1)/temp)
all = torch.exp(-similarity_matrix/temp)

loss = -(1./batch_size) * torch.sum(torch.log(torch.sum(positives, dim=1) / torch.sum(all, dim=1)))
print(loss)









# %%
# Executing training steps manually to verify that training actually works
# Testing this with different loss functions

from ionimage_embedding.models.crl.cnnClust import CNNClust
from ionimage_embedding.dataloader.crl_data import CRLdata

def get_data():
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

    clrdat = CRLdata(ds_list, test=0.3, val=0.1, 
                 cache=True, cache_folder='/scratch/model_testing',
                 colocml_preprocessing=True, 
                 fdr=.1, batch_size=100, transformations=None)
    
    return clrdat


dat = get_data()

# Plot images the test images
tstids = np.array([[  0,   1,   2,   3,   4,   5],
                   [100, 101, 102,  99,  98,  97]])

# %%
def plot_images(tstids):
    fig, axs = plt.subplots(tstids.shape[0], tstids.shape[1])

    for r in range(tstids.shape[0]):
        for c in range(tstids.shape[1]):
            axs[r, c].imshow(dat.train_dataset.images[tstids[r, c]])
            axs[r, c].axis('off')
    plt.show()

plot_images(tstids)

# %%
# Create torch dataset
from ionimage_embedding.dataloader.utils import pairwise_same_elements
from ionimage_embedding.models.coloc.utils import torch_cosine

idx = torch.tensor(tstids.reshape((-1)))
# Add dimension (required for CNN)
images = torch.tensor(dat.train_dataset.images[tstids.reshape((-1))])[:, None, :, :]
ds_labels = dat.train_dataset.dataset_labels[idx]

ion_labels = torch.arange(len(idx))
ion_labels[2]= 100
ion_labels[8]= 100
ion_label_mat = torch.tensor(pairwise_same_elements(ion_labels).astype(int))

# Image_coloc
colocs = torch_cosine(images)

# Move everything to device
device = 'cuda'
images = images.to(device)
idx = idx.to(device)
ds_labels = ds_labels.to(device)
ion_label_mat = ion_label_mat.to(device)
colocs = colocs.to(device)

# %%
# Function to load model and optimizer
def get_model_optimizer(height, width, activation='relu'):

    model = CNNClust(num_clust = 100, height=height, width=width, activation=activation, dropout=0)
    optim = torch.optim.RMSprop(params=model.parameters(), lr=0.01)

    return model, optim
















# %% FIRST LOSS: CRL2
class CRL2_version:
    pass

# Batch specific quantiles are not consideres here
model, optim = get_model_optimizer(height=dat.height, width=dat.width)
model = model.to(device)
bceloss = torch.nn.BCELoss()

# %%
from ionimage_embedding.models.crl.pseudo_labeling import compute_dataset_ublb

def crl2_loss(features, colocs, idx, ds_labels, ion_label_mat):
    features = functional.normalize(features, p=2, dim=-1)
    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

    dataset_ub, dataset_lb = compute_dataset_ublb(colocs, ds_labels=ds_labels,
                                                  lower_bound=30, upper_bound=60, device=device)
    
    ub_m = torch.ones(sim_mat.shape, device=device)
    lb_m = torch.zeros(sim_mat.shape, device=device)

    # Loop over all datasets
    for ds in torch.unique(ds_labels):
        ds_v = ds_labels == ds
        # Mask to subset similarities just to one dataset
        mask = torch.outer(ds_v, ds_v)

        # Create dataset specific threshold matrices
        ub_m[mask] = dataset_ub[ds]
        lb_m[mask] = dataset_lb[ds]
    
    # Apply thresholds to GROUND TRUGH COSINE 
    pos_loc = (colocs > ub_m).float()
    neg_loc = (colocs < lb_m).float()


    # Align the same ions
    ion_submat = ion_label_mat
    
    pos_loc = torch.maximum(pos_loc, ion_submat)
    neg_loc = torch.minimum(neg_loc, 1 - ion_submat)

    # Remove diagonal
    mask = torch.eye(pos_loc.size(0), dtype=torch.bool)
    pos_loc[mask] = 0.
    neg_loc[mask] = 0.

    pos = sim_mat[pos_loc==1.]
    neg = sim_mat[neg_loc==1.]
    out = torch.clip(torch.cat([pos, neg]), 0.0, 1.0)
    
    target = torch.cat([torch.ones(pos.shape[0], device=device), 
                        torch.zeros(neg.shape[0], device=device)])

    return bceloss(out, target), (pos_loc - neg_loc).detach().cpu().numpy()



# %%
fig, (ax1, ax2, ax3) = plt.subplots(3)



# Print model coloc agreement before
optim.zero_grad()
features = model(images)
sim_mat = torch_cosine(features)
tmp1 = (sim_mat - colocs).detach().cpu().numpy()
sns.heatmap((sim_mat - colocs).detach().cpu().numpy(), vmin=-1, vmax=1, ax=ax1, cmap='RdBu')

optim.zero_grad()

# Compute loss
features = model(images)
loss, mask = crl2_loss(features, colocs, idx, ds_labels, ion_label_mat)
ax1.set_title(f'Before (Loss: {loss})')
# Backpropagation
loss.backward()
optim.step()

sns.heatmap(mask, vmin=-1, vmax=1, ax=ax3, cmap='RdBu')

# Print model agreement after
optim.zero_grad()
features = model(images)
sim_mat = torch_cosine(features)
tmp2 = (sim_mat - colocs).detach().cpu().numpy()
sns.heatmap((sim_mat - colocs).detach().cpu().numpy(), vmin=-1, vmax=1, ax=ax2, cmap='RdBu')
ax2.set_title('After')
optim.zero_grad()
plt.show()
# %%

















# %% SECOND LOSS: Soft-Nearest neighbors
class SNN:
    pass

# Batch specific quantiles are not consideres here
model, optim = get_model_optimizer(height=dat.height, width=dat.width)
model = model.to(device)
bceloss = torch.nn.BCELoss()


# %% Loss function
def snn_loss(features, colocs, idx, ds_labels, ion_label_mat):

    # We need per row the number of positives and negatives

    # For this we still compute positives and negatives as before
    features = functional.normalize(features, p=2, dim=-1)
    sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

    dataset_ub, dataset_lb = compute_dataset_ublb(colocs, ds_labels=ds_labels,
                                                  lower_bound=30, upper_bound=60, device=device)
    
    ub_m = torch.ones(sim_mat.shape, device=device)
    lb_m = torch.zeros(sim_mat.shape, device=device)
    # Loop over all datasets
    for ds in torch.unique(ds_labels):
        ds_v = ds_labels == ds
        # Mask to subset similarities just to one dataset
        mask = torch.outer(ds_v, ds_v)

        # Create dataset specific threshold matrices
        ub_m[mask] = dataset_ub[ds]
        lb_m[mask] = dataset_lb[ds]
    
    # Apply thresholds to GROUND TRUGH COSINE 
    pos_loc = (colocs > ub_m).float()
    neg_loc = (colocs < lb_m).float()
    # Align the same ions
    ion_submat = ion_label_mat
    
    pos_loc = torch.maximum(pos_loc, ion_submat)
    neg_loc = torch.minimum(neg_loc, 1 - ion_submat)

    # Remove diagonal
    mask = torch.eye(pos_loc.size(0), dtype=torch.bool)
    pos_loc[mask] = 0.
    neg_loc[mask] = 0.

    pos = sim_mat[pos_loc==1.]
    neg = sim_mat[neg_loc==1.]
    
    # Dirty with a loop, but simple option for now:
    pos_loc[0]

    # TODO: Implement loss
    
    return pos_loc, neg_loc, pos, neg
    
    # temp = torch.tensor(400.)

    # positives = torch.exp(-similarity_matrix[labels.bool()].view(labels.shape[0], -1)/temp)
    # all = torch.exp(-similarity_matrix/temp)

    # loss = -(1./batch_size) * torch.sum(torch.log(torch.sum(positives, dim=1) / torch.sum(all, dim=1)))

pos_loc, neg_loc, pos, neg = snn_loss(features, colocs, idx, ds_labels, ion_label_mat)
# %%
