from typing import Literal, Dict, Tuple
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

from ...logger import DictLogger

class gnnDiscrete:

    model: gnnDiscreteModel

    def __init__(self, data: ColocNetData_discrete, latent_dims: int=10,
                 lr=1e3, encoding: Literal['onehot', 'learned']= 'onehot',
                 embedding_dims: int=10,
                 training_epochs: int = 11, lightning_device: str = 'gpu',
                 loss: Literal['recon', 'coloc'] = 'recon'
                 ) -> None:
        
        self.data = data
        self.latent_dims = latent_dims
        self.encoding: Literal['onehot', 'learned'] = encoding

        self.training_epochs = training_epochs
        self.lr = lr
        self.lightning_device = lightning_device
        self.embedding_dims = embedding_dims
        self.loss: Literal['recon', 'coloc'] = loss

    def train(self) -> DictLogger:
        self.model = gnnDiscreteModel(n_ions=self.data.n_ions,
                                      latent_dims=self.latent_dims,
                                      encoding=self.encoding,
                                      embedding_dims=self.embedding_dims,
                                      lr=self.lr, loss=self.loss)
        
        dictlogger = DictLogger()
        trainer = pl.Trainer(max_epochs=self.training_epochs, accelerator=self.lightning_device, 
                             logger=dictlogger,
                             callbacks=[EarlyStopping(monitor="Validation loss", mode="min")])
        trainer.fit(self.model, self.data.get_traindataloader(), self.data.get_valdataloader())

        return dictlogger
    
    def predict(self, data: Data) -> torch.Tensor:
        return self.model(data.x, data.edge_index)

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
        # Get set of ion labels from all data objects
        ion_labels = []
        for i in range(len(data)):
            ion_labels.extend(list(data[i].x.detach().cpu().numpy())) # type: ignore
        ion_labels = sorted(list(set(ion_labels)))

        # Predict for each ion using the unconnected graph
        pred = self.model(torch.tensor(ion_labels).long(), 
                          torch.tensor([[], []]).long()) # Empty edge_index
        
        pred = pred.detach().cpu().numpy()

        df = pd.DataFrame(pred, index=ion_labels)
        df.index.name = 'ion'
        # sort the dataframe by the index
        df = df.sort_index(inplace=False)

        return df
    
    def predict_from_union(self, data: ColocNetDiscreteDataset) -> pd.DataFrame:
        
        ion_labels, edge_index = compute_union_graph(data)

        # Predict for each ion using the unconnected graph
        pred = self.model(ion_labels, 
                          edge_index)        
        pred = pred.detach().cpu().numpy()

        df = pd.DataFrame(pred, index=ion_labels.detach().cpu().numpy())
        df.index.name = 'ion'
        # sort the dataframe by the index
        df = df.sort_index(inplace=False)

        return df