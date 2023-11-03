import unittest
import math
import numpy as np
import torch

from ionimage_embedding.dataloader.clr_data import CLRdata, CLRlods

ds_list = evaluation_datasets = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s'
                  ]

dat2 = CLRdata(ds_list, test=0.3, val=0.1, cache=True, cache_folder='/scratch/model_testing')
dat3 = CLRlods(ds_list, test=2, val=0.1, cache=True, cache_folder='/scratch/model_testing')

              
class TestCLR(unittest.TestCase):
    
    def test_lengths(self):
        
        test_check = math.floor(len(dat2.data) * dat2.test)
        val_check = math.floor((len(dat2.data)-test_check) * dat2.val)
        train_check = len(dat2.data) - test_check - val_check
        # test lengths
        self.assertEqual(len(dat2.data), len(dat2.train_dataset.images)+len(dat2.val_dataset.images)+len(dat2.test_dataset.images))
        
        self.assertEqual(len(dat2.train_dataset.images), train_check)
        self.assertEqual(len(dat2.train_dataset.dataset_labels), train_check)
        self.assertEqual(len(dat2.train_dataset.ion_labels), train_check)
        self.assertEqual(len(dat2.train_dataset.index), train_check)
        
        self.assertEqual(len(dat2.val_dataset.images), val_check)
        self.assertEqual(len(dat2.val_dataset.dataset_labels), val_check)
        self.assertEqual(len(dat2.val_dataset.ion_labels), val_check)
        self.assertEqual(len(dat2.val_dataset.index), val_check)
        
        self.assertEqual(len(dat2.test_dataset.images), test_check)
        self.assertEqual(len(dat2.test_dataset.dataset_labels), test_check)
        self.assertEqual(len(dat2.test_dataset.ion_labels), test_check)
        self.assertEqual(len(dat2.test_dataset.index), test_check)

    def test_indices(self):
        self.assertEqual(sorted(list(dat2.train_dataset.index)+list(dat2.val_dataset.index)), 
                         list(range(len(dat2.train_dataset.index)+len(dat2.val_dataset.index))))
        
    def test_shapes(self):
        self.assertEqual(dat2.train_dataset.images.shape[1], dat2.train_dataset.images.shape[2])
        
    def test_knn(self):
        self.assertTrue((np.array(dat2.knn_adj.sum(axis=1))>=np.repeat(dat2.k, dat2.knn_adj.shape[0])).all())
        
    def test_lods_ds(self):
        self.assertEqual(len(torch.unique(dat3.test_dataset.dataset_labels)), 2)
        
        for x in torch.unique(dat3.test_dataset.dataset_labels):
            self.assertFalse(x in dat3.train_dataset.dataset_labels)
            self.assertFalse(x in dat3.val_dataset.dataset_labels)
 