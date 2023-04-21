import torch
import numpy as np
import torch.nn.functional as functional
from random import sample

from .cae import CAE
from .cnnClust import CNNClust
from .pseudo_labeling import pseudo_labeling, run_knn, string_similarity_matrix


class DeepClustering(object):
    def __init__(self,
                 images: np.ndarray,
                 dataset_labels: np.ndarray,
                 ion_labels: np.ndarray,
                 num_cluster: int = 7,
                 initial_upper: int = 98,
                 initial_lower: int = 46,
                 upper_iteration: float = 1,
                 lower_iteration: float = 4,
                 dataset_specific_percentiles: bool = False,
                 knn: bool = False, k: int = 10,
                 lr: float = 0.0001, batch_size: float = 128,
                 pretraining_epochs: int = 11,
                 training_epochs: int = 11,
                 cae_encoder_dim: int = 7,
                 use_gpu: bool = True):
        super(DeepClustering, self).__init__()

        # Image data
        self.image_data = images
        self.dataset_labels = dataset_labels
        self.ion_labels = ion_labels

        # Image parameters
        self.num_cluster = num_cluster
        self.height = self.image_data.shape[1]
        self.width = self.image_data.shape[2]
        self.sampleN = len(self.image_data)

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
        self.use_gpu = use_gpu
        self.lr = lr
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.loss_func = torch.nn.MSELoss()
        self.cae_encoder_dim = cae_encoder_dim
        self.device = torch.device("cuda" if use_gpu else "cpu")

        # image normalization
        for i in range(0, self.sampleN):
            current_min = np.min(self.image_data[i, ::])
            current_max = np.max(self.image_data[i, ::])
            self.image_data[i, ::] = (self.image_data[i, ::] - current_min) / (current_max - current_min)

        if knn:
            self.knn_adj = run_knn(self.image_data.reshape((self.image_data.shape[0], -1)),
                                   k=self.k)

        self.ion_label_mat = string_similarity_matrix(self.ion_labels)

    @staticmethod
    def get_batch(train_image, batch_size, dataset_labels, ion_labels):
        sample_id = sample(range(len(train_image)), batch_size)
        # index = [[]]
        # index[0] = [x for x in range(batch_size)]
        # index.append(sample_id)
        batch_image = train_image[sample_id, ]
        batch_datasets = dataset_labels[sample_id, ]
        batch_ions = ion_labels[sample_id, ]

        return batch_image, sample_id, batch_datasets, batch_ions

    def train(self):
        
        cae = CAE(height=self.height, width=self.width, encoder_dim=self.cae_encoder_dim).to(self.device)
        clust = CNNClust(num_clust=self.num_cluster, height=self.height, width=self.width).to(self.device)
        
        model_params = list(cae.parameters()) + list(clust.parameters())
        optimizer = torch.optim.RMSprop(params=model_params, lr=0.001, weight_decay=0)
        # torch.optim.Adam(model_params, lr=lr)

        uu = self.initial_upper
        ll = self.initial_lower
        loss_list = list()

        random_seed = 1224
        torch.manual_seed(random_seed)
        if self.use_gpu:
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        # Pretraining of CAE only
        for epoch in range(0, self.pretraining_epochs):
            losses = list()
            for it in range(501):
                train_x, index, train_datasets, train_ions = self.get_batch(self.image_data,
                                                                            self.batch_size,
                                                                            self.dataset_labels,
                                                                            self.ion_labels)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))
                optimizer.zero_grad()
                x_p = cae(train_x)

                loss = self.loss_func(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print('Pretraining Epoch: {} Loss: {:.6f}'.format(
                      epoch, sum(losses)/len(losses)))

        # Todo: Why is optimizer initialized a second time?
        optimizer = torch.optim.RMSprop(params=model_params, lr=0.01, weight_decay=0.0)

        # Full model training
        for epoch in range(0, self.training_epochs):

            losses = list()
            losses2 = list()

            train_x, index, train_datasets, train_ions = self.get_batch(self.image_data,
                                                                        self.batch_size,
                                                                        self.dataset_labels,
                                                                        self.ion_labels)

            train_x = torch.Tensor(train_x).to(self.device)
            train_x = train_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(train_x)
            features = clust(x_p)
            # Normalization of clustering features
            features = functional.normalize(features, p=2, dim=-1)
            # Another normalization !?
            features = features / features.norm(dim=1)[:, None]
            # Similarity as defined in formula 2 of the paper
            sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

            for it in range(31):
                train_x, index, train_datasets, train_ions = self.get_batch(self.image_data,
                                                                            self.batch_size,
                                                                            self.dataset_labels,
                                                                            self.ion_labels)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))

                optimizer.zero_grad()
                x_p = cae(train_x)

                loss1 = self.loss_func(x_p, train_x)

                features = clust(x_p)
                # Feature normalization
                features = functional.normalize(features, p=2, dim=-1)
                features = features / features.norm(dim=1)[:, None]
                # Similarity computation as defined in formula 2 of the paper
                sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

                # Compute Dataset specific percentiles

                sim_numpy = sim_mat.cpu().detach().numpy()
                # Get all sim values from the batch excluding the diagonal
                tmp2 = [sim_numpy[i][j] for i in range(0, self.batch_size)
                        for j in range(self.batch_size) if i != j]
                # Compute upper and lower percentiles according to uu & ll
                ub = np.percentile(tmp2, uu)
                lb = np.percentile(tmp2, ll)

                pos_loc, neg_loc = pseudo_labeling(ub=ub, lb=lb, sim=sim_numpy, index=index, knn=self.KNN,
                                                   knn_adj=self.knn_adj, ion_label_mat=self.ion_label_mat)
                pos_loc = pos_loc.to(self.device)
                neg_loc = neg_loc.to(self.device)

                pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
                neg_entropy = torch.mul(-torch.log(torch.clip(1-sim_mat, 1e-10, 1)), neg_loc)

                loss2 = pos_entropy.sum()/pos_loc.sum() + neg_entropy.sum()/neg_loc.sum()

                loss = 1000*loss1 + loss2

                losses.append(loss1.item())
                losses2.append(loss2.item())
                loss.backward()
                optimizer.step()
                loss_list.append(sum(losses)/len(losses))

            uu = uu - self.upper_iteration
            ll = ll + self.lower_iteration
            print('Training Epoch: {} Loss: {:.6f}'.format(
                epoch, sum(losses) / len(losses)))
        return cae, clust

    def inference(self, cae, clust):
        with torch.no_grad():
            prediction_label = list()

            test_x = torch.Tensor(self.image_data).to(self.device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(test_x)
            pseudo_label = clust(x_p)

            pseudo_label = torch.argmax(pseudo_label, dim=1)
            prediction_label.extend(pseudo_label.cpu().detach().numpy())
            prediction_label = np.array(prediction_label)

            return prediction_label
