import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

from metaspace import SMInstance

import torch.nn.functional as functional
import torch
import torchvision.transforms as transforms
import lightning.pytorch as pl

from ionimage_embedding.dataloader.clr_dataloader import get_clr_dataloader
from ionimage_embedding.models.clr.cae import CAE
from ionimage_embedding.models.clr.clr_model import CLRmodel
from ionimage_embedding.models.clr.pseudo_labeling import run_knn, string_similarity_matrix

from .test_clr_utils import load_data

import unittest

class TestCLRlightning(unittest.TestCase):
    
    def setUp(self):
        training_data, training_datasets, training_ions, testing_data, testing_datasets, testing_ions = load_data(cache=True, cache_folder='/scratch/model_testing')
        
        # preparing dataloaders
        self.ds_encoder = preprocessing.LabelEncoder()
        self.dsl_int = torch.tensor(self.ds_encoder.fit_transform(training_datasets))
        self.il_encoder = preprocessing.LabelEncoder()
        self.ill_int = torch.tensor(self.il_encoder.fit_transform(training_ions))
        
        self.height = training_data.shape[1]
        self.width = training_data.shape[2]
        self.sampleN = len(training_data)
        self.batch_size=128
        val_data_fraction = .3
        
        # Image normalization
        for i in range(0, self.sampleN):
            current_min = np.min(training_data[i, ::])
            current_max = np.max(training_data[i, ::])
            training_data[i, ::] = (training_data[i, ::] - current_min) / (current_max - current_min)
        
        
        training_mask = np.arange(training_data.shape[0])
        val_mask = np.random.randint(training_data.shape[0], size=math.floor(training_data.shape[0] * val_data_fraction))
        training_mask = np.ones(len(training_data), bool)
        training_mask[val_mask] = 0
        
        self.train_dataloader = get_clr_dataloader(images=training_data[training_mask],
                                                   dataset_labels=self.dsl_int[training_mask],
                                                   ion_labels=self.ill_int[training_mask],
                                                   height=self.height,
                                                   width=self.width,
                                                   index=np.arange(training_data.shape[0])[training_mask],
                                                   # Rotate images
                                                   transform=transforms.RandomRotation(degrees=(0, 360)),
                                                   batch_size=self.batch_size)
        
        self.val_dataloader = get_clr_dataloader(images=training_data[val_mask],
                                                 dataset_labels=self.dsl_int[val_mask],
                                                 ion_labels=self.ill_int[val_mask],
                                                 height=self.height,
                                                 width=self.width,
                                                 index=np.arange(training_data.shape[0])[val_mask],
                                                 # Rotate images
                                                 transform=transforms.RandomRotation(degrees=(0, 360)),
                                                 batch_size=20, val=True)
        
        self.cae_model = CAE(self.height, self.width, encoder_dim=7, lr=0.01)
        
        self.random_seed = np.random.randint(0, 10000)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        
        # Get KNN if knn:
        self.knn_adj = torch.tensor(run_knn(training_data.reshape((training_data.shape[0], -1)), k=10))
            
        self.ion_label_mat = torch.tensor(string_similarity_matrix(training_ions))
    
    def get_new_batch(self, device):
        
        dl_image, dl_sample_id, dl_dataset_label, dl_ion_label = next(iter(self.train_dataloader))
        
        return (dl_image, 
                dl_sample_id.detach().reshape(-1).to(device), # dl_sample_id.cpu().detach().numpy().reshape(-1), 
                dl_dataset_label.detach().reshape(-1).to(device), #dl_dataset_label.cpu().detach().numpy().reshape(-1), 
                dl_ion_label.detach().reshape(-1).to(device))
    
#     def test_1_original_loss(self):
#         print('Testing original loss')
        
#         torch.manual_seed(self.random_seed)
        
        
#         optimizer = torch.optim.RMSprop(params=self.cae_model.parameters(), lr=0.01)
#         mse_loss = torch.nn.MSELoss()
#         for epoch in range(0, 5):
#             losses = list()
#             for it in range(100):
#                 self.cae_model.to('cuda')
#                 self.cae_model.train()
#                 train_x, index, train_datasets, train_ions = self.get_new_batch('cuda')
#                 train_x = train_x.to('cuda')
#                 optimizer.zero_grad()
#                 x_p = self.cae_model(train_x)

#                 loss = mse_loss(x_p, train_x)
#                 loss.backward()
#                 optimizer.step()
#                 losses.append(loss.item())

#             print('Pretraining Epoch: {:02d} Training Loss: {:.6f}'.format(
#                       epoch, sum(losses)/len(losses)))
        
#         self.assertTrue(True)
        
    def test_fullmodel_train(self):
        print('Test full model')
        
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=10, logger=False)
        trainer.fit(self.cae_model, self.train_dataloader, self.val_dataloader)
        
        
        model = CLRmodel(height=self.height, width=self.width, num_cluster=8, 
                         encoder_dim=7, lr=0.01, knn=True, knn_adj=self.knn_adj, 
                         ion_label_mat=self.ion_label_mat, dataset_specific_percentiles=True, 
                         cae_pretrained_model=self.cae_model, overweight_cae=1000.)
        
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=10, logger=False)
        trainer.fit(model, self.train_dataloader, self.val_dataloader)
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()