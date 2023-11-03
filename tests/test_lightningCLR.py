import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from metaspace import SMInstance

import torch.nn.functional as functional
import torch
import torchvision.transforms as transforms
import lightning.pytorch as pl

from ionimage_embedding.models.clr.cae import CAE
from ionimage_embedding.models.clr.clr_model import CLRmodel
from ionimage_embedding.dataloader.clr_data import CLRdata


import unittest

class TestCLRlightning(unittest.TestCase):
    
    def setUp(self):
        ds_list = evaluation_datasets = [
            '2022-12-07_02h13m50s',
            '2022-12-07_02h13m20s',
            '2022-12-07_02h10m45s',
            '2022-12-07_02h09m41s',
            '2022-12-07_02h08m52s'
                          ]

        self.dat = CLRdata(ds_list, test=0.3, val=0.1, cache=True, cache_folder='/scratch/model_testing')
        
        self.random_seed = np.random.randint(0, 10000)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        
    def test_fullmodel_train(self):
        print('Test full model')
        
        cae_model = CAE(self.dat.height, self.dat.width, encoder_dim=7, lr=0.01)
        
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=10, logger=False)
        trainer.fit(cae_model, self.dat.get_train_dataloader(), self.dat.get_val_dataloader())
        
        
        model = CLRmodel(height=self.dat.height, width=self.dat.width, num_cluster=8, 
                         encoder_dim=7, lr=0.01, knn=True, knn_adj=self.dat.knn_adj, 
                         ion_label_mat=self.dat.ion_label_mat, dataset_specific_percentiles=True, 
                         cae_pretrained_model=cae_model)
        
        trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=10, logger=False)
        trainer.fit(model, self.dat.get_train_dataloader(), self.dat.get_val_dataloader())
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()