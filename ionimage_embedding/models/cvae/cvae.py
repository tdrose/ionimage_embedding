import torch
import torch.nn.functional as functional
import lightning.pytorch as pl

from typing import Literal, Optional
import numpy as np

from ...dataloader.IonImage_data import IonImagedata_random
from .cvae_model import CVAEmodel
from ...logger import DictLogger

class CVAE:

    model: CVAEmodel

    def __init__(self, 
                 data: IonImagedata_random,
                 num_cluster: int = 7,
                 lr: float = 0.01,
                 training_epochs: int = 11,
                 lightning_device: str = 'gpu',
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 clip_gradients: Optional[float] = None,
                 weight_decay: float = 1e-4):
        
        # Image data
        self.data = data
        self._height = data.height
        self._width = data.width

        self.latent_dim = num_cluster


        self.lr = lr
        self.weight_decay = weight_decay
        self.training_epochs = training_epochs
        self.lightning_device = lightning_device
        self.activation: Literal['softmax', 'relu', 'sigmoid'] = activation
        self.clip_gradients = clip_gradients


        self.train_dataloader = data.get_train_dataloader()
        self.val_dataloader = data.get_val_dataloader()

    def train(self):

        self.model = CVAEmodel(height=self._height, width=self._width, 
                               n_classes=self.data.n_datasets, 
                               latent_dim=self.latent_dim, activation=self.activation,
                               lr=self.lr, weight_decay=self.weight_decay)

        dictlogger = DictLogger()

        trainer = pl.Trainer(max_epochs=self.training_epochs, accelerator=self.lightning_device, 
                             gradient_clip_val=self.clip_gradients, logger=dictlogger)

        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

        return dictlogger

    def inference_embeddings(self, new_data, ion_labels, dataset_labels, normalize_images=False, 
                             normalize_embeddings=False, device='cpu'):
          
        model = self.model
            
        with torch.no_grad():
            if normalize_images:
                new_data = self.image_normalization(new_data=new_data)
            
            test_x = torch.tensor(new_data, device=device)
            
            test_x = test_x.reshape((-1, 1, self._height, self._width)).to(device)
            
            model = model.to(device)
            model.eval()

            _x_hat, embeddings, _recon_loss, _kl_div, _same_ion_loss = model(test_x, 
                                                                             dataset_labels.to(device), 
                                                                             ion_labels.to(device))

            
            if normalize_embeddings:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

            return embeddings.cpu().detach().numpy()
        
    def inference_embeddings_train(self, device='cpu', use_embed_layer=False, 
                                   normalize_embeddings=False):
        return self.inference_embeddings(new_data=self.data.train_dataset.images,
                                         ion_labels=self.data.train_dataset.ion_labels,
                                         dataset_labels=self.data.train_dataset.dataset_labels,
                                         normalize_images=False, 
                                         normalize_embeddings=normalize_embeddings, device=device)
    
    def inference_embeddings_val(self, device='cpu', use_embed_layer=False, 
                                 normalize_embeddings=False):
        return self.inference_embeddings(new_data=self.data.val_dataset.images,
                                         ion_labels=self.data.val_dataset.ion_labels,
                                         dataset_labels=self.data.val_dataset.dataset_labels,
                                         normalize_images=False, 
                                         normalize_embeddings=normalize_embeddings, device=device)
    
    def inference_embeddings_test(self, device='cpu', use_embed_layer=False, 
                                  normalize_embeddings=False):
        return self.inference_embeddings(new_data=self.data.test_dataset.images,
                                         ion_labels=self.data.test_dataset.ion_labels,
                                         dataset_labels=self.data.test_dataset.dataset_labels,
                                         normalize_images=False, 
                                         normalize_embeddings=normalize_embeddings, device=device)
    
    def image_normalization(self, new_data: np.ndarray):
        return self.data.image_normalization(new_data)
