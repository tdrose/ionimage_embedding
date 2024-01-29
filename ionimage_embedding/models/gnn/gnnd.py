from ast import Dict
from typing import Literal
from IPython import embed

import torch
import lightning.pytorch as pl

from ...dataloader.ColocNet_data import ColocNetData_discrete
from .gnnDiscreteModel import gnnDiscreteModel

from ...logger import DictLogger

class gnnDiscrete:

    model: gnnDiscreteModel

    def __init__(self, data: ColocNetData_discrete, latent_dims: int=10,
                 lr=1e3, encoding: Literal['onehot', 'learned']= 'onehot',
                 embedding_dims: int=10,
                 training_epochs: int = 11, lightning_device: str = 'gpu'
                 ) -> None:
        
        self.data = data
        self.latent_dims = latent_dims
        self.encoding: Literal['onehot', 'learned'] = encoding

        self.training_epochs = training_epochs
        self.lr = lr
        self.lightning_device = lightning_device
        self.embedding_dims = embedding_dims


    def train(self) -> DictLogger:
        self.model = gnnDiscreteModel(n_ions=self.data.n_ions, 
                                      top_k=self.data.top_k, 
                                      latent_dims=self.latent_dims,
                                      encoding=self.encoding,
                                      embedding_dims=self.embedding_dims,
                                      lr=self.lr)
        
        dictlogger = DictLogger()
        trainer = pl.Trainer(max_epochs=self.training_epochs, accelerator=self.lightning_device, 
                             logger=dictlogger)
        trainer.fit(self.model, self.data.get_traindataloader(), self.data.get_valdataloader())

        return dictlogger
    
    # TODO: Implement prediction functions
