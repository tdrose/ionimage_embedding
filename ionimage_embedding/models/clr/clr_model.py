import torch
import numpy as np
from typing import Literal
import torch.nn.functional as functional
import torchvision.transforms as transforms
from random import sample
from sklearn import preprocessing
import math

from .cae import CAE
from .cnnClust import CNNClust
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
                 use_gpu: bool = True,
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
        self.use_gpu = use_gpu
        self.lr = lr
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.mse_loss = torch.nn.MSELoss()
        self.cae_encoder_dim = cae_encoder_dim
        self.device = torch.device("cuda" if use_gpu else "cpu")
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
            self.knn_adj = torch.tensor(run_knn(self.image_data.reshape((self.image_data.shape[0], -1)), k=self.k)).to(self.device)
            
        self.ion_label_mat = torch.tensor(string_similarity_matrix(self.ion_labels)).to(self.device)

        # Models
        self.cae = None
        self.clust = None
        self.loss_list = []
        self.val_losses_cae = []
        self.val_losses_clust = []

        # Dataloader
        if val_data_fraction <= 0:
            training_mask = np.arange(self.image_data.shape[0])
        else:
            val_mask = np.random.randint(self.image_data.shape[0],
                                         size=math.floor(self.image_data.shape[0] * val_data_fraction))
            training_mask = np.ones(len(self.image_data), bool)
            training_mask[val_mask] = 0

            self.val_x = self.image_data[val_mask]
            self.val_dsl_int = self.dsl_int[val_mask]
            self.val_ill_int = self.ill_int[val_mask]
            self.val_sample_id = val_mask
        
        self.training_dataloader = get_clr_dataloader(images=self.image_data[training_mask],
                                                      dataset_labels=self.dsl_int[training_mask],
                                                      ion_labels=self.ill_int[training_mask],
                                                      height=self.height,
                                                      width=self.width,
                                                      index=np.arange(self.image_data.shape[0])[training_mask],
                                                      # Rotate images
                                                      transform=transforms.RandomRotation(degrees=(0, 360)),
                                                      batch_size=self.batch_size)

        

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

    def get_new_batch(self):
        
        dl_image, dl_sample_id, dl_dataset_label, dl_ion_label = next(iter(self.training_dataloader))
        
        return (dl_image, 
                dl_sample_id.detach().reshape(-1).to(self.device), # dl_sample_id.cpu().detach().numpy().reshape(-1), 
                dl_dataset_label.detach().reshape(-1).to(self.device), #dl_dataset_label.cpu().detach().numpy().reshape(-1), 
                dl_ion_label.detach().reshape(-1).to(self.device)) #dl_ion_label.cpu().detach().numpy().reshape(-1))    

    def cl(self, neg_loc, pos_loc, sim_mat):
        pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
        neg_entropy = torch.mul(-torch.log(torch.clip(1 - sim_mat, 1e-10, 1)), neg_loc)

        # CNN loss
        contrastive_loss = pos_entropy.sum() / pos_loc.sum() + neg_entropy.sum() / neg_loc.sum()

        return contrastive_loss
    
    def compute_ublb(self, features, uu, ll, train_datasets, index):
        features = functional.normalize(features, p=2, dim=-1)
        features = features / features.norm(dim=1)[:, None]

        sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

        mask = torch.eye(sim_mat.size(0), dtype=torch.bool)
        masked_matrix = sim_mat[~mask]

        ub = torch.quantile(masked_matrix, uu/100).detach()
        lb = torch.quantile(masked_matrix, ll/100).detach()

        return ub, lb, sim_mat

    def contrastive_loss(self, features, uu, ll, train_datasets, index):

        ub, lb, sim_mat = self.compute_ublb(features, uu, ll, train_datasets, index)
        
        dataset_ub = None
        dataset_lb = None
        if self.dataset_specific_percentiles:
            
            dataset_ub, dataset_lb = compute_dataset_ublb(sim_mat, ds_labels=train_datasets,
                                                          lower_bound=ll, upper_bound=uu)

        pos_loc, neg_loc = pseudo_labeling(ub=ub, lb=lb, sim=sim_mat, index=index, knn=self.KNN,
                                           knn_adj=self.knn_adj, ion_label_mat=self.ion_label_mat,
                                           dataset_specific_percentiles=self.dataset_specific_percentiles,
                                           dataset_ub=dataset_ub, dataset_lb=dataset_lb,
                                           ds_labels=train_datasets, device=self.device)


        return self.cl(neg_loc, pos_loc, sim_mat)

    def initialize_models(self):
        cae = CAE(height=self.height, width=self.width, encoder_dim=self.cae_encoder_dim).to(self.device)
        clust = CNNClust(num_clust=self.num_cluster, height=self.height, width=self.width,
                         activation=self.activation, dropout=self.cnn_dropout).to(self.device)
        
        model_params = list(cae.parameters()) + list(clust.parameters())
        optimizer = torch.optim.RMSprop(params=model_params, lr=self.lr, weight_decay=self.weight_decay)
        
        torch.manual_seed(self.random_seed)
        if self.use_gpu:
            torch.cuda.manual_seed(self.random_seed)
            torch.backends.cudnn.deterministic = True # noqa
        
        return cae, clust, optimizer
    
    def train(self):
        
        cae, clust, optimizer = self.initialize_models()
        
        uu = self.initial_upper
        ll = self.initial_lower
        

        
        # Get validation data
        val_x = torch.tensor(self.val_x).reshape((-1, 1, self.height, self.width))
        val_x = val_x.to(self.device)
        
        val_losses_cae = list()
        val_losses_clust = list()
        loss_list = list()
            
        # Pretraining of CAE only
        for epoch in range(0, self.pretraining_epochs):
            losses = list()
            for it in range(100):
                cae.train()
                train_x, index, train_datasets, train_ions = self.get_new_batch()
                train_x = train_x.to(self.device)
                optimizer.zero_grad()
                x_p = cae(train_x)

                loss = self.mse_loss(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
            if self.val_data_fraction <= 0:
                val_loss = 0
            else:
                cae.eval()
                with torch.no_grad():
                    x_p = cae(val_x)
                    optimizer.zero_grad()
                    val_loss = self.mse_loss(x_p, val_x)

            print('Pretraining Epoch: {:02d} Training Loss: {:.6f} | Validation Loss: {:.6f}'.format(
                      epoch, sum(losses)/len(losses), val_loss))

        # Todo: Why is optimizer initialized a second time?
        # optimizer = torch.optim.RMSprop(params=model_params, lr=self.lr, weight_decay=self.weight_decay)

        # Full model training
        for epoch in range(0, self.training_epochs):
            
            # CAE loss
            losses_cae = list()
            # CNN loss
            losses_clust = list()
            # combined loss
            loss_combined = list()

            cae.train()
            clust.train()
            for it in range(31):
                
                train_x, index, train_datasets, train_ions = self.get_new_batch()
                train_x = train_x.to(self.device)

                optimizer.zero_grad()
                x_p = cae(train_x)

                loss_cae = self.mse_loss(x_p, train_x)
                features = clust(x_p)

                loss_clust = self.contrastive_loss(features=features, uu=uu, ll=ll,
                                                   train_datasets=train_datasets, index=index)

                loss = self.overweight_cae*loss_cae + loss_clust

                losses_cae.append(loss_cae.item())
                losses_clust.append(loss_clust.item())
                loss_combined.append(loss.item())
                
                loss.backward()
                if self.clip_gradients is not None:
                    torch.nn.utils.clip_grad_value_(model_params, clip_value=self.clip_gradients)
                optimizer.step()
                loss_list.append(sum(loss_combined)/len(loss_combined))
            
            if self.val_data_fraction <= 0:
                val_cael = 0
                val_cnnl = 0
            else:
                cae.eval()
                clust.eval()
                with torch.no_grad():
                    x_p = cae(val_x)
                    optimizer.zero_grad()
                    val_cael = self.mse_loss(x_p, val_x)
                    features = clust(x_p)
                    val_cnnl = self.contrastive_loss(features=features, uu=uu, ll=ll,
                                                     train_datasets=self.val_dsl_int, index=self.val_sample_id)

                    val_losses_cae.append(float(val_cael))
                    val_losses_clust.append(float(val_cnnl))
                
            uu = uu - self.upper_iteration
            ll = ll + self.lower_iteration
            
            print('Epoch: {:02d} | CAE-Loss: {:.6f} | CNN-Loss: {:.6f} | Total loss: {:.6f}'.format(
                                                                        epoch,
                                                                        sum(losses_cae) / len(losses_cae),
                                                                        sum(losses_clust) / len(losses_clust),
                                                                        sum(loss_combined) / len(loss_combined))
                  )
            print('  * Val:  | CAE-Loss: {:.6f} | CNN-Loss: {:.6f}'.format(val_cael, val_cnnl))
            
        self.loss_list = loss_list
        self.val_losses_cae = val_losses_cae
        self.val_losses_clust = val_losses_clust
        
        self.cae = cae
        self.clust = clust
        
        return 0

    def inference(self, cae=None, clust=None, new_data: np.ndarray = None):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clust
        
        with torch.no_grad():
            prediction_label = list()

            if new_data is None:
                test_x = torch.Tensor(self.image_data).to(self.device)
                
            else:
                nd = self.image_normalization(new_data=new_data)
                test_x = torch.Tensor(nd).to(self.device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(test_x)
            pseudo_label = clust(x_p)

            pseudo_label = torch.argmax(pseudo_label, dim=1)
            prediction_label.extend(pseudo_label.cpu().detach().numpy())
            prediction_label = np.array(prediction_label)

            return prediction_label

    def predict_embeddings(self, cae=None, clust=None, new_data: np.ndarray = None, normalize=True):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clust
            
        with torch.no_grad():
            if new_data is None:
                test_x = torch.Tensor(self.image_data).to(self.device)
            else:
                nd = self.image_normalization(new_data=new_data)
                test_x = torch.Tensor(nd).to(self.device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(test_x)
            embeddings = clust(x_p)
            
            if normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)
                embeddings = embeddings / embeddings.norm(dim=1)[:, None]

            return embeddings.cpu().detach().numpy()
