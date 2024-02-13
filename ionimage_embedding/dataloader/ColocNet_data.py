from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from warnings import warn

import torch
from torch_geometric.loader import DataLoader

from .IonImage_data import IonImagedata_random
from ..coloc.coloc import ColocModel
from .utils import cache_hashing
from .constants import COLOC_NET_DISCRETE_DATA
from .ColocNetDiscreteDataset import ColocNetDiscreteDataset


class ColocNetData_discrete:

    _test_set: np.ndarray
    _val_set: np.ndarray
    _train_set: np.ndarray

    def __init__(self, dataset_ids: List[str], test: int=1, val: int=1,
                 top_k: int=3, bottom_k: int=3,
                 db: Tuple[str, str]=('HMDB', 'v4'), fdr: float=0.2, scale_intensity: str='TIC', 
                 colocml_preprocessing: bool=False, min_images: int=6, maxzero: float=.95,
                 batch_size: int=128,
                 cache_images: bool=False, cache_folder: str='/scratch/model_testing', 
                 random_network: bool=False,
                 use_precomputed: bool=False,
                 ion_labels: Optional[torch.Tensor]=None,
                 ds_labels: Optional[torch.Tensor]=None,
                 coloc: Optional[Dict[int, pd.DataFrame]]=None,
                 dsl_int_mapper: Optional[Dict[int, str]]=None,
                 ion_int_mapper: Optional[Dict[int, str]]=None,
                 force_reload: bool=False
                ) -> None:
        
        if min_images < top_k + bottom_k:
            raise ValueError('min_images must larger or equal to top_k + bottom_k')
        
        if random_network:
            warn('A random network is used. This network does not contain meaningful information.')
        
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.dataset_ids = dataset_ids
        self.db = db
        self.fdr = fdr
        self.scale_intensity = scale_intensity
        self.colocml_preprocessing = colocml_preprocessing
        self.min_images = min_images
        self.maxzero = maxzero
        self.random_network = random_network


        cache_hex = cache_hashing(dataset_ids, colocml_preprocessing, db, fdr, scale_intensity,
                                  maxzero=maxzero, vitb16_compatible=False, force_size=False, 
                                  min_images=min_images)
        
        if force_reload:
            # In torch_gemetric <=2.4.0 force reload is not implemented for inmemory datasets
            self.dataset_file = '{}_{}_{}_{}_{}_{}'.format(COLOC_NET_DISCRETE_DATA, 
                                                           cache_hex, 
                                                           str(top_k), str(bottom_k),
                                                           str(random_network),
                                                           str(np.random.randint(0, 1000000)))
        else:
            self.dataset_file = '{}_{}_{}_{}_{}'.format(COLOC_NET_DISCRETE_DATA, 
                                                    cache_hex, 
                                                    str(top_k), str(bottom_k),
                                                    str(random_network))
        
        self.batch_size = batch_size
        
        if not use_precomputed:
            # Create the IonImagedata object otherwise use the given one
            iidata = IonImagedata_random(
                dataset_ids=dataset_ids, test=.3, val=.1, db=db, fdr=fdr,
                scale_intensity=scale_intensity, 
                colocml_preprocessing=colocml_preprocessing,
                batch_size=batch_size, cache=cache_images, 
                cache_folder=cache_folder,
                min_images=min_images, maxzero=maxzero)
        
            colocs = ColocModel(iidata)

            # Extract the list of unique ion labels from the colocs
            

            ion_labels = iidata.full_dataset.ion_labels
            ds_labels = iidata.full_dataset.dataset_labels
            coloc = colocs.full_coloc

            self.dsl_int_mapper = iidata.dsl_int_mapper
            self.ion_int_mapper = iidata.ion_int_mapper

        else:
            if ion_labels is None or ds_labels is None or coloc is None or \
                dsl_int_mapper is None or ion_int_mapper is None:
                
                raise ValueError('If use_precomputed is True, ion_labels, '
                                 'ds_labels and coloc must be provided.')

            self.dsl_int_mapper = dsl_int_mapper
            self.ion_int_mapper = ion_int_mapper

        self.n_ions = ion_labels.unique().shape[0]

        self.dataset = ColocNetDiscreteDataset(path=cache_folder,
                                               name=self.dataset_file,
                                               top_k=self.top_k, bottom_k=self.bottom_k,
                                               min_ion_count=min_images,
                                               ion_labels=ion_labels,
                                               ds_labels=ds_labels,
                                               coloc=coloc, 
                                               random_network=random_network)

        print(self.dataset)
        print({k: v.shape for k, v in self.dataset.coloc_dict.items()})
        
        self.update_test_val(test, val)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def update_test_val(self, test: int, val: int) -> None:
        if test + val > len(self.dataset):
            raise ValueError('test and val must be smaller than dataset size')
        self.test = test
        self.val = val
        self.sample_sets()
    
    def updated_k_data(self, top_k: int, bottom_k: int):
        if top_k <= self.top_k and bottom_k <= self.bottom_k:
            raise ValueError('The new k values must be larger than the current ones.')
        
        dat = ColocNetData_discrete(
            dataset_ids=self.dataset_ids,
            test=self.test, val=self.val,
            top_k=top_k, bottom_k=bottom_k,
            db=self.db, fdr=self.fdr, scale_intensity=self.scale_intensity, 
            colocml_preprocessing=self.colocml_preprocessing, min_images=self.min_images, 
            maxzero=self.maxzero,
            batch_size=self.batch_size,
            cache_images=False, cache_folder='/scratch/model_testing', 
            random_network=self.random_network,
            use_precomputed=True, # Important
            ion_labels=self.dataset.ion_labels,
            ds_labels=self.dataset.ds_labels,
            coloc=self.dataset.coloc_dict,
            dsl_int_mapper=self.dsl_int_mapper,
            ion_int_mapper=self.ion_int_mapper
            )
        
        # Copy the sets
        dat._test_set = self._test_set
        dat._val_set = self._val_set
        dat._train_set = self._train_set
        
        return dat

    def sample_sets(self) -> None:
        mask = np.random.choice(np.arange(len(self.dataset)), self.test, replace=False)
        tmp = np.delete(np.arange(len(self.dataset)), mask)
        
        self._test_set = mask
        self._val_set = np.random.choice(tmp, self.val, replace=False)
        
        train_mask = np.array([i not in self._test_set and i not in self._val_set 
                               for i in np.arange(len(self.dataset))])
        
        self._train_set = np.arange(len(self.dataset))[train_mask]
    
    
    def get_traindataloader(self):
        return DataLoader(self.dataset.index_select(self._train_set), 
                          batch_size=self.batch_size, shuffle=True) 

    def get_testdataloader(self):
        return DataLoader(self.dataset.index_select(self._test_set), 
                          batch_size=self.batch_size, shuffle=False)

    def get_valdataloader(self):
        return DataLoader(self.dataset.index_select(self._val_set), 
                          batch_size=self.batch_size, shuffle=False)
    
    def get_train_dsids(self) -> torch.Tensor:
        return self.dataset.index_select(self._train_set).ds_label.unique() # type: ignore
    
    def get_test_dsids(self) -> torch.Tensor:
        return self.dataset.index_select(self._test_set).ds_label.unique() # type: ignore
    
    def get_val_dsids(self) -> torch.Tensor:
        return self.dataset.index_select(self._val_set).ds_label.unique() # type: ignore


class MeanColocNetData_discrete:

    _test_set: np.ndarray
    _val_set: np.ndarray
    _train_set: np.ndarray

    def __init__(self, dataset_ids: List[str], test: int=1, val: int=1,
                 top_k: int=3, bottom_k: int=3,
                 db: Tuple[str, str]=('HMDB', 'v4'), fdr: float=0.2, scale_intensity: str='TIC', 
                 colocml_preprocessing: bool=False, min_images: int=6, maxzero: float=.95,
                 batch_size: int=128,
                 cache_images: bool=False, cache_folder: str='/scratch/model_testing',
                ) -> None:
        if min_images < top_k + bottom_k:
            raise ValueError('min_images must larger or equal to top_k + bottom_k')
        
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.dataset_ids = dataset_ids
        self.db = db
        self.fdr = fdr
        self.scale_intensity = scale_intensity
        self.colocml_preprocessing = colocml_preprocessing
        self.min_images = min_images
        self.maxzero = maxzero
        self.cache_folder = cache_folder


        cache_hex = cache_hashing(dataset_ids, colocml_preprocessing, db, fdr, scale_intensity,
                                  maxzero=maxzero, vitb16_compatible=False, force_size=False, 
                                  min_images=min_images)
        
        self.dataset_file = '{}_{}_{}_{}_{}'.format(COLOC_NET_DISCRETE_DATA, 
                                                 cache_hex, 
                                                 str(top_k), str(bottom_k),
                                                 str(False))
        
        self.batch_size = batch_size
        
        # Create the IonImagedata object otherwise use the given one
        iidata = IonImagedata_random(
            dataset_ids=dataset_ids, test=.3, val=.1, db=db, fdr=fdr,
            scale_intensity=scale_intensity, 
            colocml_preprocessing=colocml_preprocessing,
            batch_size=batch_size, cache=cache_images, 
            cache_folder=cache_folder,
            min_images=min_images, maxzero=maxzero)
    
        colocs = ColocModel(iidata)

        # Extract the list of unique ion labels from the colocs
        ion_labels = iidata.full_dataset.ion_labels
        ds_labels = iidata.full_dataset.dataset_labels
        coloc = colocs.full_coloc

        self.dsl_int_mapper = iidata.dsl_int_mapper
        self.ion_int_mapper = iidata.ion_int_mapper

        self.n_ions = ion_labels.unique().shape[0]


        # Train, val and test dataset need to be created differently
        # than in the ColocNetData_discrete

        # Select datasets for test
        test_mask = np.random.choice(np.arange(len(coloc)), test, replace=False)
        tmp = np.delete(np.arange(len(coloc)), test_mask)
        
        val_mask = np.random.choice(tmp, val, replace=False)
        
        train_mask = np.array([i not in test_mask and i not in val_mask 
                               for i in np.arange(len(coloc))])
        
        train_mask = np.arange(len(coloc))[train_mask]

        # Mask ds_labels and ion_labels for the train set
        
        ds_mask= self.masking(train_mask, ds_labels)
        tmp_ion_labels = ion_labels[ds_mask]

        mean_coloc, _ = colocs._inference(
            tmp_ion_labels, agg='mean', 
            ds_colocs={k: v for k, v in coloc.items() if k in train_mask})
        
        self.train_dataset = ColocNetDiscreteDataset(
            path=self.cache_folder,
            name=self.dataset_file + '_Mean_train_' + '-'.join([str(i) for i in sorted(train_mask)]),
            top_k=self.top_k, bottom_k=self.bottom_k,
            min_ion_count=self.min_images,
            ion_labels=torch.tensor(np.array(mean_coloc.index)),
            ds_labels=torch.tensor([train_mask[0]]),
            coloc={train_mask[0]: mean_coloc}, 
            random_network=False
            )

        self.val_dataset = self.create_dataset(val_mask, ion_labels, 
                                               ds_labels, coloc, '_Mean_val_')
        self.test_dataset = self.create_dataset(test_mask, ion_labels, 
                                                ds_labels, coloc, '_Mean_test_')

        self._test_set = test_mask
        self._val_set = val_mask
        self._train_set = train_mask
        
    def masking(self, mask, ds_labels):
        ds_mask = np.zeros(len(ds_labels), dtype=bool)
        for i in np.arange(len(ds_labels)):
            ds_mask[i] = i in mask

        return ds_mask
    
    def create_dataset(self, mask, ion_labels, ds_labels, coloc, name):

        ds_mask, ion_mask = self.masking(mask, ds_labels)

        return ColocNetDiscreteDataset(
            path=self.cache_folder,
            name=self.dataset_file + name + '-'.join([str(i) for i in sorted(mask)]),
            top_k=self.top_k, bottom_k=self.bottom_k,
            min_ion_count=self.min_images,
            ion_labels=ion_labels[ds_mask],
            ds_labels=ds_labels[ds_mask],
            coloc={k: v for k, v in coloc.items() if k in mask}, 
            random_network=False)
    
    
    def get_traindataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, shuffle=True) 

    def get_testdataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, shuffle=False)

    def get_valdataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, shuffle=False)
