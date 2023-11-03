import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

from .utils import download_data

class CLRdata:
    
    def __init__(self, dataset_ids, test=0.3, val=0.1, transformations=None, maindata_class = True,
                 # Download parameters:
                 db=("HMDB", "v4"), fdr=0.2, scale_intensity='TIC', hotspot_clipping=False
                ):
        
        self.dataset_ids = dataset_ids
        self._maindata_class = maindata_class
        self.transformations = transformations
        self.val = val
        self.test = test
        
        # Download data
        data, dataset_labels, ion_labels = download_data(ds_ids, db=db, fdr=fdr, scale_intensity=scale_intensity, 
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
        
        if maindata_class:
            # check if val and test data proportions contain at least a few images
            self.test_check(val=self.val, test=self.test)
            
            # Conditional - KNN
            self.check_dataloader()
    
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
        train_check = len(self.data)-test_check
        
        if test_check < min_images or val_check < min_images or train_check < min_images:
            raise ValueError(f'Test/Validation/Training data must contain at least {min_images}\n'
                             f'Train contains: {train_check}, Val contains: {val_check}, Test contains: {test_check}')
            
    def check_dataloader(self):
        if self.train_dataset is None or self.test_dataset is None or self.val_dataset is None:
            raise AttributeError('Datasets not defined, DO NOT mess with the data classes!')