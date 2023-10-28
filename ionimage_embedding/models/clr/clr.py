import torch
import numpy as np
from typing import Literal
import torch.nn.functional as functional
import torchvision.transforms as transforms
import lightning.pytorch as pl

from random import sample
from sklearn import preprocessing
import math

from .cae import CAE
from .clr_model import CLRmodel
from .pseudo_labeling import pseudo_labeling, \
    run_knn, \
    string_similarity_matrix, \
    compute_dataset_ublb
from .utils import flip_images
from ...dataloader.clr_dataloader import get_clr_dataloader


class CLR:
    def __init__(self,
                 images: np.ndarray,
                 dataset_labels: np.ndarray,
                 ion_labels: np.ndarray,
                 val_data_fraction: float = 0.2,
                 num_cluster: int = 7,
                 initial_upper: int = 98,
                 initial_lower: int = 46,
                 upper_iteration: float = 1,
                 lower_iteration: float = 4,
                 dataset_specific_percentiles: bool = False,
                 random_flip: bool = False,
                 knn: bool = False, k: int = 10,
                 lr: float = 0.01, batch_size: int = 128,
                 pretraining_epochs: int = 11,
                 training_epochs: int = 11,
                 cae_encoder_dim: int = 7,
                 lightning_device: str = 'gpu',
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 clip_gradients: float = None,
                 overweight_cae: float = 1000,
                 cnn_dropout: float = 0.1,
                 weight_decay: float = 1e-4,
                 random_seed: int = 1234):

        # Image data
        self.image_data = images
        self.dataset_labels = dataset_labels
        self.ion_labels = ion_labels
        self.val_data_fraction = val_data_fraction
        
        self.ds_encoder = preprocessing.LabelEncoder()
        self.dsl_int = torch.tensor(self.ds_encoder.fit_transform(self.dataset_labels))

        self.il_encoder = preprocessing.LabelEncoder()
        self.ill_int = torch.tensor(self.il_encoder.fit_transform(self.ion_labels))

        # Image parameters
        self.num_cluster = num_cluster
        self.height = self.image_data.shape[1]
        self.width = self.image_data.shape[2]
        self.sampleN = len(self.image_data)

        if random_flip:
            if self.height != self.width:
                raise ValueError('random_transpose only possible if image height and width are equal.')
            else:
                self.random_flip = True

        # Pseudo labeling parameters
        self.initial_upper = initial_upper
        self.initial_lower = initial_lower
        self.upper_iteration = upper_iteration
        self.lower_iteration = lower_iteration
        self.dataset_specific_percentiles = dataset_specific_percentiles

        # KNN parameters
        self.KNN = knn
        self.k = k
        self.knn_adj = None

        # Pytorch parameters
        self.activation = activation
        self.lr = lr
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.mse_loss = torch.nn.MSELoss()
        self.cae_encoder_dim = cae_encoder_dim
        self.lightning_device = lightning_device
        self.random_seed = random_seed
        self.clip_gradients = clip_gradients
        self.overweight_cae = overweight_cae
        self.cnn_dropout = cnn_dropout
        self.weight_decay = weight_decay

        print(f'After {self.training_epochs} epochs, the upper bound will be: '
              f'{self.initial_upper - (self.training_epochs * self.upper_iteration)}.')
        print(f'After {self.training_epochs} epochs, the lower bound will be: '
              f'{self.initial_lower + (self.training_epochs * self.lower_iteration)}.')

        if (self.initial_lower + (self.training_epochs * self.lower_iteration)) >= \
                (self.initial_upper - (self.training_epochs * self.upper_iteration)):
            raise ValueError(f'Lower percentile will be higher than upper percentile parameter '
                             f'after {self.training_epochs} epochs.\n'
                             f'Change initial_upper, initial_lower, upper_iteration, lower_iteration, '
                             f'or training_epochs parameters.')

        # image normalization
        self.image_normalization()

        if knn:
            self.knn_adj = torch.tensor(run_knn(self.image_data.reshape((self.image_data.shape[0], -1)), k=self.k))
            
        self.ion_label_mat = torch.tensor(string_similarity_matrix(self.ion_labels))

        # Models
        self.cae = None
        self.clust = None
        self.loss_list = []
        self.val_losses_cae = []
        self.val_losses_clust = []

        # Dataloader
        if val_data_fraction <= 0:
            raise ValueError('Validation data fraction needs to be greater than 0')
        
        val_mask = np.random.randint(self.image_data.shape[0],
                                     size=math.floor(self.image_data.shape[0] * val_data_fraction))
        training_mask = np.ones(len(self.image_data), bool)
        training_mask[val_mask] = 0
        
        self.train_dataloader = get_clr_dataloader(images=self.image_data[training_mask],
                                                   dataset_labels=self.dsl_int[training_mask],
                                                   ion_labels=self.ill_int[training_mask],
                                                   height=self.height,
                                                   width=self.width,
                                                   index=np.arange(self.image_data.shape[0])[training_mask],
                                                   transform=transforms.RandomRotation(degrees=(0, 360)),
                                                   batch_size=self.batch_size)
        
        self.val_dataloader = get_clr_dataloader(images=self.image_data[val_mask],
                                                 dataset_labels=self.dsl_int[val_mask],
                                                 ion_labels=self.ill_int[val_mask],
                                                 height=self.height,
                                                 width=self.width,
                                                 index=np.arange(self.image_data.shape[0])[val_mask],
                                                 transform=transforms.RandomRotation(degrees=(0, 360)),
                                                 batch_size=20, val=True)

        # Placeholders for models
        self.cae = None
        self.clr = None

    def image_normalization(self, new_data: np.ndarray = None):
        if new_data is None:
            for i in range(0, self.sampleN):
                current_min = np.min(self.image_data[i, ::])
                current_max = np.max(self.image_data[i, ::])
                self.image_data[i, ::] = (self.image_data[i, ::] - current_min) / (current_max - current_min)

        else:
            nd = new_data.copy()
            for i in range(0, nd.shape[0]):
                current_min = np.min(nd[i, ::])
                current_max = np.max(nd[i, ::])
                nd[i, ::] = (nd[i, ::] - current_min) / (current_max - current_min)

            return nd

    def train(self, logger=False):
        
        # Pretraining of CAE model
        cae = CAE(self.height, self.width, encoder_dim=self.cae_encoder_dim, lr=self.lr)
        
        trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, max_epochs=self.pretraining_epochs, logger=logger)
        trainer.fit(cae, self.train_dataloader, self.val_dataloader)
        
        self.cae = cae
        
        # Training of full model
        self.clr = CLRmodel(height=self.height, width=self.width, num_cluster=self.num_cluster, ion_label_mat=self.ion_label_mat, activation=self.activation, encoder_dim=self.cae_encoder_dim, 
                            initial_upper=self.initial_upper, initial_lower=self.initial_lower, upper_iteration=self.upper_iteration, lower_iteration=self.lower_iteration,
                            dataset_specific_percentiles=self.dataset_specific_percentiles, lr=self.lr, cae_pretrained_model=cae, knn=self.KNN, knn_adj = self.knn_adj, 
                            cnn_dropout=self.cnn_dropout, weight_decay=self.weight_decay, clip_gradients=self.clip_gradients)
        
        trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, max_epochs=self.training_epochs, logger=logger)
        trainer.fit(self.clr, self.train_dataloader, self.val_dataloader)
        
        return 0

    def inference(self, cae=None, clr=None, new_data: np.ndarray = None, device='cpu'):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clr
        
        with torch.no_grad():
            prediction_label = list()

            if new_data is None:
                test_x = torch.Tensor(self.image_data, device=device)
                
            else:
                nd = self.image_normalization(new_data=new_data)
                test_x = torch.Tensor(nd, device=device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))
            
            cae = cae.to(device)
            clust = clust.to(device)
            
            pseudo_label, x_p = clust(x_p)

            pseudo_label = torch.argmax(pseudo_label, dim=1)
            prediction_label.extend(pseudo_label.cpu().detach().numpy())
            prediction_label = np.array(prediction_label)

            return prediction_label

    def predict_embeddings(self, cae=None, clust=None, new_data: np.ndarray = None, normalize=True, device='cpu'):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clust
            
        with torch.no_grad():
            if new_data is None:
                test_x = torch.Tensor(self.image_data, device=device)
            else:
                nd = self.image_normalization(new_data=new_data)
                test_x = torch.Tensor(nd, device=device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))
            
            cae = cae.to(device)
            clust = clust.to(device)

            embeddings, x_p = clust(x_p)
            
            if normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)
                embeddings = embeddings / embeddings.norm(dim=1)[:, None]

            return embeddings.cpu().detach().numpy()
