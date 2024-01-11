import numpy as np
from sklearn import preprocessing
import math
from typing import Optional, List, Tuple

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

from .utils import pairwise_same_elements, run_knn, get_data
from .mzImageDataset import mzImageDataset


class IonImagedata_random:
    
    def __init__(self, dataset_ids: List[str], test: float=0.3, val: float=0.1, 
                 transformations: Optional[torch.nn.Module]=T.RandomRotation(degrees=(0, 360)), 
                 maindata_class: bool=True,
                 db: Tuple[str, str]=('HMDB', 'v4'), fdr: float=0.2, scale_intensity: str='TIC', 
                 colocml_preprocessing: bool=False,
                 k: int=10, batch_size: int=128,
                 cache: bool=False, cache_folder: str='/scratch/model_testing', min_images: int=5, 
                 maxzero: float=.95
                ):
        
        self.dataset_ids = dataset_ids
        self._maindata_class = maindata_class
        self.transformations = transformations
        self.val = val
        self.test = test
        self.k=k
        self.batch_size=batch_size
        self.min_images=min_images
        
        # Get data
        data, dataset_labels, ion_labels = get_data(dataset_ids=dataset_ids, cache=cache, 
                                                    cache_folder=cache_folder, db=db, fdr=fdr, 
                                                    scale_intensity=scale_intensity, 
                                                    colocml_preprocessing=colocml_preprocessing,
                                                    maxzero=maxzero)
        
        # Normalize images
        data = self.image_normalization(data)
        
        # Encoding for ds_labels and ion_labels
        self.ds_encoder = preprocessing.LabelEncoder()
        dsl_int = torch.tensor(self.ds_encoder.fit_transform(dataset_labels))
        
        self.dsl_int_mapper = {}
        for dsid in set(dataset_labels):
            mask = dataset_labels == dsid
            idx = np.arange(len(dataset_labels))[mask][0]
            self.dsl_int_mapper[dsl_int.detach().cpu().numpy()[idx]] = dsid
        

        self.il_encoder = preprocessing.LabelEncoder()
        ill_int = torch.tensor(self.il_encoder.fit_transform(ion_labels))
        
        # self.data = data
        print(data.shape)
        self.height = data.shape[1]
        self.width = data.shape[2]

        self.full_dataset = mzImageDataset(images=data, 
                                           dataset_labels=dsl_int,
                                           ion_labels=ill_int,
                                           height=self.height,
                                           width=self.width,
                                           index=np.arange(len(data)),
                                           transform=None)
        
        if maindata_class:
            # check if val and test data proportions contain at least a few images
            self.check_val_test_proportions(val=self.val, test=self.test, 
                                            min_images=self.min_images)
            
            # Train test split
            test_mask = np.random.choice(self.full_dataset.images.shape[0], 
                                         size=math.floor(self.full_dataset.images.shape[0] * \
                                                         self.test), 
                                         replace=False)
            tmp_mask = np.ones(len(self.full_dataset.images), bool)
            tmp_mask[test_mask] = 0
            tmp = self.split_data(mask=tmp_mask, data=self.full_dataset.images, 
                                  dsl=self.full_dataset.dataset_labels, 
                                  ill=self.full_dataset.ion_labels)
            tmp_data, tmp_dls, tmp_ill, tmp_index, test_data, test_dls, test_ill, test_index = tmp
            
            # Train val split
            val_mask = np.random.choice(tmp_data.shape[0], 
                                        size=math.floor(tmp_data.shape[0] * self.val), 
                                        replace=False)
            train_mask = np.ones(len(tmp_data), bool)
            train_mask[val_mask] = 0
            tmp = self.split_data(mask=train_mask, data=tmp_data, dsl=tmp_dls, ill=tmp_ill)
            train_data, train_dls, train_ill, train_idx, val_data, val_dls, val_ill, val_idx = tmp
            
            # compute KNN and ion_label_mat (For combined train/val data)
            self.knn_adj: torch.Tensor = torch.tensor(run_knn(tmp_data.reshape((tmp_data.shape[0], 
                                                                                -1)), k=self.k))
            
            self.ion_label_mat: torch.Tensor = torch.tensor(pairwise_same_elements(tmp_ill
                                                                                   ).astype(int))
            
            # Make datasets
            self.train_dataset = mzImageDataset(images=train_data, 
                                                dataset_labels=train_dls,
                                                ion_labels=train_ill,
                                                height=self.height,
                                                width=self.width,
                                                index=train_idx,
                                                transform=self.transformations)
            
            self.val_dataset = mzImageDataset(images=val_data, 
                                              dataset_labels=val_dls,
                                              ion_labels=val_ill,
                                              height=self.height,
                                              width=self.width,
                                              index=val_idx,
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
        
        return (split1_data, split1_dls, split1_ill, split1_index, split2_data, 
                split2_dls, split2_ill, split2_index)
    
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, drop_last=True)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset.images), 
                          shuffle=False, drop_last=True)
    
    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset.images), 
                          shuffle=False, drop_last=True)
    
    # simple min-max scaling per image, no data leakage
    def image_normalization(self, new_data: np.ndarray):
                nd = new_data.copy()
                for i in range(0, nd.shape[0]):
                    current_min = np.min(nd[i, ::])
                    current_max = np.max(nd[i, ::])
                    if current_min != 0 or current_max != 0:
                        nd[i, ::] = (nd[i, ::] - current_min) / (current_max - current_min)
                return nd
    
    # Will be overwritten by inheriting classes
    def check_val_test_proportions(self, val, test, min_images=5):
        if not self._maindata_class:
            raise AttributeError('Method cannot be executed if inherited data class is used')
        
        test_check = math.floor(len(self.full_dataset.images) * self.test)
        val_check = math.floor((len(self.full_dataset.images)-test_check) * self.val)
        train_check = len(self.full_dataset.images)-test_check - val_check
        
        if test_check < min_images or val_check < min_images or train_check < min_images:
            raise ValueError(f'Test/Validation/Training data must contain at least {min_images}\n'
                             f'Train contains: {train_check}, Val contains: {val_check},'
                             f'Test contains: {test_check}')
            
    def check_dataexists(self):
        if self.train_dataset is None or self.test_dataset is None or self.val_dataset is None:
            raise AttributeError('Datasets not defined, DO NOT mess with the data classes!')
            
            
class IonImagedata_leaveOutDataSet(IonImagedata_random):
    
    def __init__(self, dataset_ids, test=1, val=0.1, 
                 transformations=T.RandomRotation(degrees=(0, 360)),
                 # Download parameters:
                 db=('HMDB', 'v4'), fdr=0.2, scale_intensity='TIC', colocml_preprocessing=False,
                 k=10, batch_size=128,
                 cache=False, cache_folder='/scratch/model_testing', min_images=5, 
                 maxzero: float=.95
                ):
        
        if test < 1:
            raise ValueError('CLRlods (leave datasets out) class requires value for test larger '
                             'than 1 (number of datasets excluded from training)')
            
        super().__init__(dataset_ids=dataset_ids, test=.1, val=val, transformations=transformations, 
                         maindata_class=False,
                         db=db, fdr=fdr, scale_intensity=scale_intensity, 
                         colocml_preprocessing=colocml_preprocessing,
                         k=k, batch_size=batch_size,
                         cache=cache, cache_folder=cache_folder, min_images=min_images, 
                         maxzero=maxzero)
        
        # Train test split
        if len(self.dataset_ids) <= test:
            raise ValueError('Cannot assing more datasets for testing that loaded datasets')
        
        test_dsid = np.random.choice(torch.unique(self.full_dataset.dataset_labels).numpy(), 
                                     size=test, replace=False)
        tmp_mask = np.ones(len(self.full_dataset.images), bool)
        for ds in test_dsid:
            tmp_mask[self.full_dataset.dataset_labels==ds] = 0
        tmp = self.split_data(mask=tmp_mask, data=self.full_dataset.images, 
                              dsl=self.full_dataset.dataset_labels, 
                              ill=self.full_dataset.ion_labels)
        tmp_data, tmp_dls, tmp_ill, tmp_index, test_data, test_dls, test_ill, test_index = tmp
    
        
        # Train val split
        val_mask = np.random.choice(tmp_data.shape[0], size=math.floor(tmp_data.shape[0] * \
                                                                       self.val), replace=False)
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
        
        
class IonImagedata_transitivity(IonImagedata_random):
    
    def __init__(self, dataset_ids, test=.3, val=0.1, 
                 transformations=T.RandomRotation(degrees=(0, 360)),
                 # Download parameters:
                 db=('HMDB', 'v4'), fdr=0.2, scale_intensity='TIC', colocml_preprocessing=False,
                 k=10, batch_size=128,
                 cache=False, cache_folder='/scratch/model_testing', 
                 min_images=5, min_codetection=2, maxzero: float=.95
                ):
            
        super().__init__(dataset_ids=dataset_ids, test=.3, val=val, 
                         transformations=transformations, maindata_class=False,
                         db=db, fdr=fdr, scale_intensity=scale_intensity, 
                         colocml_preprocessing=colocml_preprocessing,
                         k=k, batch_size=batch_size,
                         cache=cache, cache_folder=cache_folder, min_images=min_images, 
                         maxzero=maxzero)
        
        self.min_codetection=min_codetection
        
        # Train test split
        index_dict = self.codetection_index(ill=self.full_dataset.ion_labels, 
                                            dsl=self.full_dataset.dataset_labels, 
                                            min_codetection=self.min_codetection)
        tmp_mask = self.codetection_mask(idx_dict=index_dict, test_fraction=test, 
                                         ill=self.full_dataset.ion_labels, 
                                         dsl=self.full_dataset.dataset_labels)
        tmp = self.split_data(mask=tmp_mask, data=self.full_dataset.images, 
                              dsl=self.full_dataset.dataset_labels, 
                              ill=self.full_dataset.ion_labels)
        tmp_data, tmp_dls, tmp_ill, tmp_index, test_data, test_dls, test_ill, test_index = tmp
    
        
        # Train val split
        val_mask = np.random.choice(tmp_data.shape[0], size=math.floor(tmp_data.shape[0] * \
                                                                       self.val), replace=False)
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
    
    @staticmethod
    def codetection_index(ill, dsl, min_codetection=2):
        
        index_dict = {}
        
        for dsid in torch.unique(dsl):
            mask = dsl==dsid
            
            masked_ill = ill[mask]
            
            for x in range(len(masked_ill)):
                for y in range(x+1, len(masked_ill)):
                    tmp = tuple(sorted([int(masked_ill[x]),int(masked_ill[y])]))
                    
                    if tmp not in index_dict.keys():
                        index_dict[tmp] = [int(dsid)]
                    else:
                        if int(dsid) not in index_dict[tmp]:
                            index_dict[tmp].append(int(dsid))
        
        # Return ion pairs that are co-detected in at least 2 datasets
        return {k: v for k, v in index_dict.items() if len(v)>=min_codetection}

    @staticmethod
    def codetection_mask(idx_dict, test_fraction, ill, dsl):
        
        # Because numpy.choice is not possible on list of tuples
        idx_keys = list(idx_dict.keys())
        idx_int = np.arange(len(idx_keys))
        
        tmp = np.random.choice(idx_int, size=math.floor(len(idx_int) * test_fraction), 
                               replace=False)
        tmp = [idx_keys[x] for x in tmp]
        
        
        out_mask = np.ones(len(ill), bool)
        
        # For each ion pair
        for ip in tmp:
            
            ds = idx_dict[ip]
            # Make random selection of which ion to remove for all datasets
            ids = np.random.choice(list(ip), size=len(ds), replace=True)
            
            # For each dataset where ions co-occur, Search for array position
            for x in range(len(ds)):
                a = (ill==ids[x])&(dsl==ds[x])
                pos = int(torch.arange(len(a))[a][0])
                
                out_mask[pos] = False
                
        return out_mask
