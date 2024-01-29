from fileinput import filename
from typing import List, Tuple, Optional
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from .IonImage_data import IonImagedata_random
from ..coloc.coloc import ColocModel
from .utils import cache_hashing
from .constants import COLOC_NET_DISCRETE_DATA
from .ColocNetDiscreteDataset import ColocNetDiscreteDataset


class ColocNetData_discrete:
    def __init__(self, dataset_ids: List[str], test: int=1, val: int=1,
                 top_k: int=3,
                 db: Tuple[str, str]=('HMDB', 'v4'), fdr: float=0.2, scale_intensity: str='TIC', 
                 colocml_preprocessing: bool=False, min_images: int=6, maxzero: float=.95,
                 batch_size: int=128,
                 cache_images: bool=False, cache_folder: str='/scratch/model_testing'
                ) -> None:
        
        if min_images > 2*top_k:
            raise ValueError('min_images must be smaller than 2 times top_k')
        
        iidata = IonImagedata_random(dataset_ids=dataset_ids, test=.3, val=.1, db=db, fdr=fdr,
                                      scale_intensity=scale_intensity, 
                                      colocml_preprocessing=colocml_preprocessing,
                                      batch_size=batch_size, cache=cache_images, 
                                      cache_folder=cache_folder,
                                      min_images=min_images, maxzero=maxzero)
        
        colocs = ColocModel(iidata)

        # Extract the list of unique ion labels from the colocs
        self.n_ions = iidata.full_dataset.ion_labels.unique().shape[0]

        self.top_k = top_k

        cache_hex = cache_hashing(dataset_ids, colocml_preprocessing, db, fdr, scale_intensity,
                                  maxzero=maxzero, vitb16_compatible=False, force_size=False)
        
        self.dataset_file = '{}_{}_{}'.format(COLOC_NET_DISCRETE_DATA, 
                                                 cache_hex, str(top_k))


        self.dataset = ColocNetDiscreteDataset(path=cache_folder,
                                               name=self.dataset_file,
                                               top_k=self.top_k,
                                               ion_labels=iidata.full_dataset.ion_labels,
                                               ds_labels=iidata.full_dataset.dataset_labels,
                                               coloc=colocs.full_coloc)

        self.batch_size = batch_size

        if test + val > len(self.dataset):
            raise ValueError('test and val must be smaller than dataset size')
        
        mask = np.random.choice(np.arange(len(self.dataset)), test)
        tmp = np.delete(np.arange(len(self.dataset)), mask)
        
        self._test_set = mask
        self._val_set = np.random.choice(tmp, val, replace=False)
        
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
