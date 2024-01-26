from fileinput import filename
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data

from .IonImage_data import IonImagedata_random
from ..models import ColocModel
from .utils import cache_hashing
from .constants import COLOC_NET_DISCRETE_DATA
from .ColocNetDiscreteDataset import ColocNetDiscreteDataset




# Creats a torch_geometric.data.Data object for each coloc graph
# Creates positive and negative edge list that is also used for evaluation
# Holds the dataset object and function to return the dataloader

class ColocNetData_discrete:
    def __init__(self, dataset_ids: List[str], test: float=0.3, val: float=0.1,
                 top_k: int=3,
                 db: Tuple[str, str]=('HMDB', 'v4'), fdr: float=0.2, scale_intensity: str='TIC', 
                 colocml_preprocessing: bool=False, min_images: int=6, maxzero: float=.95,
                 batch_size: int=128,
                 cache_images: bool=False, cache_folder: str='/scratch/model_testing'
                ) -> None:
        
        if min_images > 2*top_k:
            raise ValueError('min_images must be smaller than 2 times top_k')
        
        iidata = IonImagedata_random(dataset_ids=dataset_ids, test=test, val=val, db=db, fdr=fdr,
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

        # Divide data into train and test data
        # TODO


    def get_dataloader(self):
        pass