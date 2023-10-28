import numpy as np
import pandas as pd

import torch.nn.functional as functional
import torch

from ionimage_embedding.models import CLR
from ionimage_embedding.models.clr.pseudo_labeling import compute_dataset_ublb, pseudo_labeling
from ionimage_embedding.models.clr.utils import size_adaption, size_adaption_symmetric

from .test_clr_utils import original_ublb, original_dataset_ublb, original_ps, load_data

import unittest

# ############################
# Preprocessing - Loading data
# ############################
print('##############')
print('Starting download of data and preprocessing')
print('##############')

training_data, training_datasets, training_ions, testing_data, testing_datasets, testing_ions = load_data(cache=True, cache_folder='/scratch/model_testing')

model = CLR(
            images=training_data,
            dataset_labels=training_datasets,
            ion_labels=training_ions,
            num_cluster=8,
            initial_upper=93,
            initial_lower=37,
            upper_iteration=1.5,
            lower_iteration=1.5,
            dataset_specific_percentiles=True,
            random_flip=True,
            knn=True, k=5,
            lr=0.0001, batch_size=128,
            pretraining_epochs=1,
            training_epochs=2,
            cae_encoder_dim=2,
            lightning_device='gpu',
            random_seed=1225
            )

device='cuda'

model.train(logger=False)

cae = model.cae.to(device)
clust = model.clr.to(device)

optimizer = torch.optim.RMSprop(params=clust.parameters(), lr=model.lr, weight_decay=model.weight_decay)

train_x, index, train_datasets, train_ions = next(iter(model.train_dataloader))

train_x = train_x.to(device)
train_datasets = train_datasets.reshape(-1)
train_ions = train_ions.reshape(-1)



optimizer.zero_grad()
features, x_p = clust(train_x)
uu = 85
ll=55



print('##############')
print('Preprocessing done')
print('##############')

class TestCLR(unittest.TestCase):
    
    def setUp(self):
        self.oub, self.olb, self.sim_numpy, self.smn = original_ublb(model, features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        self.tub, self.tlb, self.sim_mat = clust.compute_ublb(features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        
        self.ods_ub, self.ods_lb = original_dataset_ublb(self.sim_numpy, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)
        self.ods_ubv = [self.ods_ub[x] for x in range(len(self.ods_ub))]
        self.ods_lbv = [self.ods_lb[x] for x in range(len(self.ods_lb))]

        self.tds_ub, self.tds_lb = compute_dataset_ublb(self.sim_mat, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)

    def test_1(self):
        self.assertTrue(self.tub > self.tlb)
        self.assertTrue(self.oub > self.olb)
        self.assertEqual(self.oub, self.tub)
        self.assertEqual(self.olb, self.tlb)

    def test_2(self):
        self.assertTrue(all((self.tds_ub - self.tds_lb) >= 0))
        self.assertTrue(all([np.round(float(self.tds_ub[x].cpu().numpy()), 3)==np.round(self.ods_ubv[x], 3) for x in range(self.tds_ub.size(0))]))
        self.assertTrue(all([np.round(float(self.tds_lb[x].cpu().numpy()), 3)==np.round(self.ods_lbv[x], 3) for x in range(self.tds_lb.size(0))]))
        
    def test_3(self):
        
        # no KNN, no DS percentiles
        opl, onl = original_ps(ub=self.oub, lb=self.olb, sim=self.sim_numpy, index=index, knn=False, knn_adj=model.knn_adj.cpu().numpy(), ion_label_mat=model.ion_label_mat.cpu().numpy(),
                               dataset_specific_percentiles=False, dataset_ub=self.ods_ub, dataset_lb=self.ods_lb, ds_labels=train_datasets)
        self.assertFalse(((opl.cpu().numpy() + onl.cpu().numpy())==2).any())
        
        tpl, tnl = pseudo_labeling(ub=self.tub, lb=self.tlb, sim=self.sim_mat, index=index, knn=False, knn_adj=model.knn_adj.to(device), ion_label_mat=model.ion_label_mat.to(device), dataset_specific_percentiles=False,
                                   dataset_ub=self.tds_ub, dataset_lb=self.tds_lb, ds_labels=train_datasets, device=device)
        self.assertFalse(((tpl.cpu().numpy() + tnl.cpu().numpy())==2).any())
        self.assertTrue((tpl.cpu().numpy() == opl.cpu().numpy()).all())
        self.assertTrue((tnl.cpu().numpy() == onl.cpu().numpy()).all())

        # no KNN, DS percentiles
        opl, onl = original_ps(ub=self.oub, lb=self.olb, sim=self.sim_numpy, index=index, knn=False, knn_adj=model.knn_adj.cpu().numpy(), ion_label_mat=model.ion_label_mat.cpu().numpy(),
                               dataset_specific_percentiles=True, dataset_ub=self.ods_ub, dataset_lb=self.ods_lb, ds_labels=train_datasets)
        self.assertFalse(((opl.cpu().numpy() + onl.cpu().numpy())==2).any())
        
        tpl, tnl = pseudo_labeling(ub=self.tub, lb=self.tlb, sim=self.sim_mat, index=index, knn=False, knn_adj=model.knn_adj.to(device), ion_label_mat=model.ion_label_mat.to(device), dataset_specific_percentiles=True,
                                   dataset_ub=self.tds_ub, dataset_lb=self.tds_lb, ds_labels=train_datasets, device=device)
        self.assertFalse(((tpl.cpu().numpy() + tnl.cpu().numpy())==2).any())
        self.assertTrue((tpl.cpu().numpy() == opl.cpu().numpy()).all())
        self.assertTrue((tnl.cpu().numpy() == onl.cpu().numpy()).all())
        
        # KNN, no DS percentiles
        opl, onl = original_ps(ub=self.oub, lb=self.olb, sim=self.sim_numpy, index=index, knn=True, knn_adj=model.knn_adj.cpu().numpy(), ion_label_mat=model.ion_label_mat.cpu().numpy(),
                               dataset_specific_percentiles=False, dataset_ub=self.ods_ub, dataset_lb=self.ods_lb, ds_labels=train_datasets)
        self.assertFalse(((opl.cpu().numpy() + onl.cpu().numpy())==2).any())
        
        tpl, tnl = pseudo_labeling(ub=self.tub, lb=self.tlb, sim=self.sim_mat, index=index, knn=True, knn_adj=model.knn_adj.to(device), ion_label_mat=model.ion_label_mat.to(device), dataset_specific_percentiles=False,
                                   dataset_ub=self.tds_ub, dataset_lb=self.tds_lb, ds_labels=train_datasets, device=device)
        self.assertFalse(((tpl.cpu().numpy() + tnl.cpu().numpy())==2).any())
        self.assertTrue((tpl.cpu().numpy() == opl.cpu().numpy()).all())
        self.assertTrue((tnl.cpu().numpy() == onl.cpu().numpy()).all())
        
        # KNN, DS percentiles
        opl, onl = original_ps(ub=self.oub, lb=self.olb, sim=self.sim_numpy, index=index, knn=True, knn_adj=model.knn_adj.cpu().numpy(), ion_label_mat=model.ion_label_mat.cpu().numpy(),
                               dataset_specific_percentiles=True, dataset_ub=self.ods_ub, dataset_lb=self.ods_lb, ds_labels=train_datasets)
        self.assertFalse(((opl.cpu().numpy() + onl.cpu().numpy())==2).any())
        
        tpl, tnl = pseudo_labeling(ub=self.tub, lb=self.tlb, sim=self.sim_mat, index=index, knn=True, knn_adj=model.knn_adj.to(device), ion_label_mat=model.ion_label_mat.to(device), dataset_specific_percentiles=True,
                                   dataset_ub=self.tds_ub, dataset_lb=self.tds_lb, ds_labels=train_datasets, device=device)
        self.assertFalse(((tpl.cpu().numpy() + tnl.cpu().numpy())==2).any())
        self.assertTrue((tpl.cpu().numpy() == opl.cpu().numpy()).all())
        self.assertTrue((tnl.cpu().numpy() == onl.cpu().numpy()).all())

    def test_4(self):
        # Original
        optimizer.zero_grad()
        oub, olb, sim_numpy, smn = original_ublb(model, features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        ods_ub, ods_lb = original_dataset_ublb(sim_numpy, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)
        opl, onl = original_ps(ub=oub, lb=olb, sim=self.sim_numpy, index=index, knn=True, knn_adj=model.knn_adj.cpu().numpy(), ion_label_mat=model.ion_label_mat.cpu().numpy(),
                               dataset_specific_percentiles=True, dataset_ub=ods_ub, dataset_lb=ods_lb, ds_labels=train_datasets)
        l1 = clust.cl(opl.to(device), onl.to(device), sim_mat = smn)
        l1.backward(retain_graph=True)
        g1 = torch.gradient(smn)[0].cpu().detach().numpy()

        # Torch
        optimizer.zero_grad()
        tub, tlb, sim_mat =  clust.compute_ublb(features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        tds_ub, tds_lb = compute_dataset_ublb(sim_mat, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)
        tpl, tnl = pseudo_labeling(ub=tub, lb=tlb, sim=self.sim_mat, index=index, knn=True,
                               knn_adj=model.knn_adj.to(device), ion_label_mat=model.ion_label_mat.to(device),
                               dataset_specific_percentiles=True,
                               dataset_ub=tds_ub, dataset_lb=tds_lb,
                               ds_labels=train_datasets, device=device)

        l2 = clust.cl(tpl, tnl, sim_mat = sim_mat)
        l2.backward(retain_graph=True)
        g2 = torch.gradient(sim_mat)[0].cpu().detach().numpy()

        self.assertTrue(l1 == l2)
        self.assertTrue((g1 == g2).all())

        
if __name__ == '__main__':
    unittest.main()
