from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from warnings import warn

import torch
from torch_geometric.loader import DataLoader

from .IonImage_data import IonImagedata_random
from .get_coloc_model import get_coloc_model
from .utils import cache_hashing
from ..constants import COLOC_NET_DISCRETE_DATA
from ..torch_datasets.ColocNetDiscreteDataset import ColocNetDiscreteDataset


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
                 ion_composition: Optional[torch.Tensor]=None,
                 coloc: Optional[Dict[int, pd.DataFrame]]=None,
                 dsl_int_mapper: Optional[Dict[int, str]]=None,
                 ion_int_mapper: Optional[Dict[int, str]]=None,
                 atom_mapper: Optional[Dict[str, int]]=None,
                 n_ions: Optional[int]=None,
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
                cache_folder=cache_folder, knn=False,
                min_images=min_images, maxzero=maxzero)
        
            colocs = get_coloc_model(iidata)

            # Extract the list of unique ion labels from the colocs
            
            ion_labels = iidata.full_dataset.ion_labels
            ds_labels = iidata.full_dataset.dataset_labels
            ion_comp = iidata.full_dataset.ion_composition
            coloc = colocs.full_coloc

            self.dsl_int_mapper = iidata.dsl_int_mapper
            self.ion_int_mapper = iidata.ion_int_mapper
            self.atom_mapper = iidata.atom_mapper

            
            self.n_ions = ion_labels.unique().shape[0]

        else:
            if ion_labels is None or ds_labels is None or coloc is None or \
               dsl_int_mapper is None or ion_int_mapper is None or \
               n_ions is None or atom_mapper is None or ion_composition is None:
                
                raise ValueError('If use_precomputed is True, ion_labels, '
                                 'ds_labels, coloc, atom_mapper, and ion_composition'
                                 ' must be provided.')

            self.dsl_int_mapper = dsl_int_mapper
            self.ion_int_mapper = ion_int_mapper
            self.atom_mapper = atom_mapper
            self.n_ions = n_ions
        
        self.atom_dims = ion_comp.shape[1]

        self.dataset = ColocNetDiscreteDataset(path=cache_folder,
                                               name=self.dataset_file,
                                               top_k=self.top_k, bottom_k=self.bottom_k,
                                               min_ion_count=min_images,
                                               ion_labels=ion_labels,
                                               ds_labels=ds_labels,
                                               ion_composition=ion_comp,
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
            ion_composition=self.dataset.ion_composition,
            coloc=self.dataset.coloc_dict,
            dsl_int_mapper=self.dsl_int_mapper,
            ion_int_mapper=self.ion_int_mapper,
            atom_mapper=self.atom_mapper
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
