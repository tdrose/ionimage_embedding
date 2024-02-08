from typing import Literal, Dict, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch_geometric.data import Data

from ...dataloader.ColocNet_data import ColocNetData_discrete
from ...dataloader.ColocNetDiscreteDataset import ColocNetDiscreteDataset
from .gnnDiscreteModel import gnnDiscreteModel
from .utils import compute_union_graph
from ..constants import VALIDATION_LOSS

from ...logger import DictLogger

class gnnDiscrete:

    model: Optional[gnnDiscreteModel] = None

    def __init__(self, data: ColocNetData_discrete, latent_dims: int=10,
                 lr=1e3, encoding: Literal['onehot', 'learned']= 'onehot',
                 embedding_dims: int=10,
                 training_epochs: int = 11, early_stopping_patience: int = 5,
                 lightning_device: str = 'gpu',
                 loss: Literal['recon', 'coloc'] = 'recon',
                 activation: Literal['softmax', 'relu', 'sigmoid', 'none']='none',
                 num_layers: int=2,
                 gnn_layer_type: Literal['GCNConv', 'GATv2Conv', 'GraphConv'] = 'GCNConv'
                 ) -> None:
        
        self.data = data
        self.latent_dims = latent_dims
        self.encoding: Literal['onehot', 'learned'] = encoding
        self.num_layers = num_layers

        self.training_epochs = training_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr = lr
        self.lightning_device = lightning_device
        self.embedding_dims = embedding_dims
        self.loss: Literal['recon', 'coloc'] = loss
        self.activation: Literal['softmax', 'relu', 'sigmoid', 'none'] = activation
        self.gnn_layer_type: Literal['GCNConv', 'GATv2Conv', 'GraphConv'] = gnn_layer_type

    def train(self) -> DictLogger:
        self.model = gnnDiscreteModel(n_ions=self.data.n_ions,
                                      latent_dims=self.latent_dims,
                                      encoding=self.encoding,
                                      embedding_dims=self.embedding_dims,
                                      lr=self.lr, loss=self.loss, 
                                      activation=self.activation,
                                      num_layers=self.num_layers,
                                      gnn_layer_type=self.gnn_layer_type)
        
        dictlogger = DictLogger()
        trainer = pl.Trainer(max_epochs=self.training_epochs, accelerator=self.lightning_device, 
                             logger=dictlogger, enable_checkpointing=False,
                             callbacks=[EarlyStopping(monitor=VALIDATION_LOSS, 
                                                      mode="min", 
                                                      patience=self.early_stopping_patience)])
        trainer.fit(self.model, self.data.get_traindataloader(), self.data.get_valdataloader())

        return dictlogger
    
    def check_model(self) -> None:
        if self.model is None:
            raise ValueError('Model has not been trained')
        
    def predict(self, data: Data) -> torch.Tensor:
        self.check_model()
        return self.model(data.x, data.edge_index, data.edge_attr) # type: ignore
    
    def fine_tune(self, data: ColocNetData_discrete, training_epochs: int = 11) -> DictLogger:
        self.check_model()

        dictlogger = DictLogger()
        trainer = pl.Trainer(max_epochs=training_epochs, accelerator=self.lightning_device, 
                             logger=dictlogger, enable_checkpointing=False,
                             callbacks=[EarlyStopping(monitor=VALIDATION_LOSS, 
                                                      mode="min", 
                                                      patience=self.early_stopping_patience)])
        
        trainer.fit(self.model, data.get_traindataloader(), data.get_valdataloader()) # type: ignore

        return dictlogger

    def predict_multiple(self, data: ColocNetDiscreteDataset) -> Dict[int, torch.Tensor]:
        out = {}
        for i in range(len(data)):
            out[i] = self.predict(data[i])# type: ignore

        return out
    
    def predict_centroids(self, data: ColocNetDiscreteDataset) -> Tuple[np.ndarray, np.ndarray]:
        pred_dict = self.predict_multiple(data)

        # Get set of ion labels from all data objects
        ion_labels = []
        for i in range(len(data)):
            ion_labels.extend(list(data[i].x.detach().cpu().numpy())) # type: ignore
        ion_labels = list(set(ion_labels))

        # Get mean prediction for each ion
        ion_centroids = []
        centroid_labels = []
        for i in ion_labels:
            tmp = []
            for dsid, pred in pred_dict.items():
                if i in data[dsid].x: # type: ignore
                    tmp.append(pred[data[dsid].x == i].detach().cpu().numpy()[0]) # type: ignore
            if len(tmp) > 1:
                a = np.stack(tmp)
                ion_centroids.append(np.mean(a, axis=0))
                centroid_labels.append(i)
            elif len(tmp) == 1:
                ion_centroids.append(tmp[0])
                centroid_labels.append(i)
            else:
                pass
        
        return np.stack(ion_centroids), np.array(centroid_labels)

    def predict_centroids_df(self, data: ColocNetDiscreteDataset) -> pd.DataFrame:
        ion_centroids, centroid_labels = self.predict_centroids(data)

        df = pd.DataFrame(ion_centroids, index=centroid_labels)
        df.index.name = 'ion'
        # sort the dataframe by the index
        df = df.sort_index(inplace=False)

        return df
    
    def predict_from_unconnected(self, data: ColocNetDiscreteDataset) -> pd.DataFrame:
        self.check_model()

        # Get set of ion labels from all data objects
        ion_labels = []
        for i in range(len(data)):
            ion_labels.extend(list(data[i].x.detach().cpu().numpy())) # type: ignore
        ion_labels = sorted(list(set(ion_labels)))

        # Predict for each ion using the unconnected graph
        pred = self.model(torch.tensor(ion_labels).long(), # type: ignore
                          torch.tensor([[], []]).long()) # Empty edge_index
        
        pred = pred.detach().cpu().numpy()

        df = pd.DataFrame(pred, index=ion_labels)
        df.index.name = 'ion'
        # sort the dataframe by the index
        df = df.sort_index(inplace=False)

        return df
    
    def predict_from_union(self, data: ColocNetDiscreteDataset) -> pd.DataFrame:
        self.check_model()

        ion_labels, edge_index = compute_union_graph(data)

        # Predict for each ion using the unconnected graph
        pred = self.model(ion_labels, # type: ignore
                          edge_index)        
        pred = pred.detach().cpu().numpy()

        df = pd.DataFrame(pred, index=ion_labels.detach().cpu().numpy())
        df.index.name = 'ion'
        # sort the dataframe by the index
        df = df.sort_index(inplace=False)

        return df