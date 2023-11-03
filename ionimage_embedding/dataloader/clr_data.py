import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
import os
import pickle

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

from .utils import download_data, pairwise_same_elements, run_knn
from .clr_dataloader import mzImageDataset


class CLRdata:
    
    def __init__(self, dataset_ids, test=0.3, val=0.1, transformations=T.RandomRotation(degrees=(0, 360)), maindata_class = True,
                 # Download parameters:
                 db=('HMDB', 'v4'), fdr=0.2, scale_intensity='TIC', hotspot_clipping=False,
                 k=10, batch_size=128,
                 cache=False, cache_folder='/scratch/model_testing'
                ):
        
        self.dataset_ids = dataset_ids
        self._maindata_class = maindata_class
        self.transformations = transformations
        self.val = val
        self.test = test
        self.k=k
        self.batch_size=batch_size
        
        # Download data
        if cache:
            # make hash of datasets
            cache_file = 'CLRdata_{}.pickle'.format(''.join(dataset_ids))

            # Check if cache folder exists
            if not os.path.isdir(cache_folder):
                os.mkdir(cache_folder)

            # Download data if it does not exist
            if cache_file not in os.listdir(cache_folder):
                data, dataset_labels, ion_labels = download_data(dataset_ids, db=db, fdr=fdr, scale_intensity=scale_intensity, 
                                                                 hotspot_clipping=hotspot_clipping)

                pickle.dump((data, dataset_labels, ion_labels), open(os.path.join(cache_folder, cache_file), "wb"))
                print('Saved file: {}'.format(os.path.join(cache_folder, cache_file)))      
            
            # Load cached data
            else:
                print('Loading cached data from: {}'.format(os.path.join(cache_folder, cache_file)))
                data, dataset_labels, ion_labels = pickle.load(open(os.path.join(cache_folder, cache_file), "rb" ) )

        else:
            data, dataset_labels, ion_labels = download_data(dataset_ids, db=db, fdr=fdr, scale_intensity=scale_intensity, 
                                                             hotspot_clipping=hotspot_clipping)
        
        
        # Normalize images
        data = self.image_normalization(data)
        
        # Encoding for ds_labels and ion_labels
        self.ds_encoder = preprocessing.LabelEncoder()
        self.dsl_int = torch.tensor(self.ds_encoder.fit_transform(dataset_labels))
        self.il_encoder = preprocessing.LabelEncoder()
        self.ill_int = torch.tensor(self.il_encoder.fit_transform(ion_labels))
        
        self.data = data
        
        # dataloader variables
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.height = data.shape[1]
        self.width = data.shape[2]
        
        if maindata_class:
            # check if val and test data proportions contain at least a few images
            self.check_val_test_proportions(val=self.val, test=self.test)
            
            # Train test split
            test_mask = np.random.choice(self.data.shape[0], size=math.floor(self.data.shape[0] * self.test), replace=False)
            tmp_mask = np.ones(len(self.data), bool)
            tmp_mask[test_mask] = 0
            tmp = self.split_data(mask=tmp_mask, data=self.data, dsl=self.dsl_int, ill=self.ill_int)
            tmp_data, tmp_dls, tmp_ill, tmp_index, test_data, test_dls, test_ill, test_index = tmp
            
            # Train val split
            val_mask = np.random.choice(tmp_data.shape[0], size=math.floor(tmp_data.shape[0] * self.val), replace=False)
            train_mask = np.ones(len(tmp_data), bool)
            train_mask[val_mask] = 0
            tmp = self.split_data(mask=train_mask, data=tmp_data, dsl=tmp_dls, ill=tmp_ill)
            train_data, train_dls, train_ill, train_index, val_data, val_dls, val_ill, val_index = tmp
            
            # compute KNN and ion_label_mat (For combined train/val data)
            self.knn_adj = torch.tensor(run_knn(tmp_data.reshape((tmp_data.shape[0], -1)), k=self.k))
            
            self.ion_label_mat = torch.tensor(pairwise_same_elements(tmp_ill).astype(int))
            
            # Make datasets
            self.train_dataset = mzImageDataset(images=train_data, 
                                                dataset_labels=train_dls,
                                                ion_labels=train_ill,
                                                height=self.height,
                                                width=self.width,
                                                index=train_index,
                                                transform=self.transformations)
            
            self.val_dataset = mzImageDataset(images=val_data, 
                                              dataset_labels=val_dls,
                                              ion_labels=val_ill,
                                              height=self.height,
                                              width=self.width,
                                              index=val_index,
                                              transform=self.transformations)
            
            self.test_dataset = mzImageDataset(images=test_data, 
                                               dataset_labels=test_dls,
                                               ion_labels=test_ill,
                                               height=self.height,
                                               width=self.width,
                                               index=test_index,
                                               transform=self.transformations)
            
            self.check_dataexists()
    
    def split_data(self, mask, data, dsl, ill):
        
        reverse_mask = ~mask
        n_samples = len(data)
        
        split1_data = data[mask]
        split1_dls = dsl[mask]
        split1_ill = ill[mask]
        split1_index = np.arange(n_samples)[mask]
        
        split2_data = data[reverse_mask]
        split2_dls = dsl[reverse_mask]
        split2_ill = ill[reverse_mask]
        split2_index = np.arange(n_samples)[reverse_mask]
        
        return split1_data, split1_dls, split1_ill, split1_index, split2_data, split2_dls, split2_ill, split2_index
    
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset.images), shuffle=False)
    
    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset.images), shuffle=False)
    
    # simple min-max scaling, no data leakage
    def image_normalization(self, new_data: np.ndarray):
                nd = new_data.copy()
                for i in range(0, nd.shape[0]):
                    current_min = np.min(nd[i, ::])
                    current_max = np.max(nd[i, ::])
                    nd[i, ::] = (nd[i, ::] - current_min) / (current_max - current_min)
                return nd
    
    # Will be overwritten by inheriting classes
    def check_val_test_proportions(self, val, test, min_images=5):
        
        test_check = math.floor(len(self.data) * self.test)
        val_check = math.floor((len(self.data)-test_check) * self.val)
        train_check = len(self.data)-test_check - val_check
        
        if test_check < min_images or val_check < min_images or train_check < min_images:
            raise ValueError(f'Test/Validation/Training data must contain at least {min_images}\n'
                             f'Train contains: {train_check}, Val contains: {val_check}, Test contains: {test_check}')
            
    def check_dataexists(self):
        if self.train_dataset is None or self.test_dataset is None or self.val_dataset is None:
            raise AttributeError('Datasets not defined, DO NOT mess with the data classes!')