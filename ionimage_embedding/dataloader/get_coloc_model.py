from ..coloc.coloc import ColocModel
from .IonImage_data import IonImagedata_random

def get_coloc_model(model: IonImagedata_random, device: str='cpu') -> ColocModel:
    
    return ColocModel(full_dataset=model.full_dataset,
                      train_dataset=model.train_dataset,
                      val_dataset=model.val_dataset,
                      test_dataset=model.test_dataset,
                      device=device)
