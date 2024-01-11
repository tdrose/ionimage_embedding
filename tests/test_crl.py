import numpy as np
import pandas as pd

import torch.nn.functional as functional
import torch

from ionimage_embedding.models import CRL
from ionimage_embedding.models.crl.pseudo_labeling import compute_dataset_ublb, pseudo_labeling
from ionimage_embedding.dataloader.utils import size_adaption, size_adaption_symmetric
from ionimage_embedding.dataloader.IonImage_data import IonImagedata_random
from ionimage_embedding.dataloader.utils import pairwise_same_elements

from .test_crl_utils import original_ublb, original_dataset_ublb, original_ps, load_data

import unittest

# ############################
# Preprocessing - Loading data
# ############################
ds_list = evaluation_datasets = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s'
                  ]

dat = IonImagedata_random(ds_list, test=0.3, val=0.1, cache=True, cache_folder='/scratch/model_testing')

model = CRL(dat,
            num_cluster=8,
            initial_upper=93,
            initial_lower=37,
            upper_iteration=1.5,
            lower_iteration=1.5,
            dataset_specific_percentiles=True,
            knn=True, 
            lr=0.0001,
            pretraining_epochs=5,
            training_epochs=5,
            cae_encoder_dim=2,
            lightning_device='gpu'
            )

device='cuda'

model.train(logger=False)

cae = model.cae.to(device)
clust = model.crl.to(device)

optimizer = torch.optim.RMSprop(params=clust.parameters(), lr=model.lr, weight_decay=model.weight_decay)

train_x, index, train_datasets, train_ions = next(iter(model.train_dataloader))

train_x = train_x.to(device)
train_datasets = train_datasets.reshape(-1)
train_ions = train_ions.reshape(-1)



optimizer.zero_grad()
features, x_p = clust(train_x)
uu = 85
ll = 55

class TestCLR(unittest.TestCase):
    
    def setUp(self):
        self.oub, self.olb, self.sim_numpy, self.smn = original_ublb(dat.batch_size, features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        self.tub, self.tlb, self.sim_mat = clust.compute_ublb(features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
        
        self.ods_ub, self.ods_lb = original_dataset_ublb(self.sim_numpy, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)
        self.ods_ubv = [self.ods_ub[x] for x in range(len(self.ods_ub))]
        self.ods_lbv = [self.ods_lb[x] for x in range(len(self.ods_lb))]

        self.tds_ub, self.tds_lb = compute_dataset_ublb(self.sim_mat, ds_labels=train_datasets, lower_bound=ll, upper_bound=uu)

    def test_1(self):
        self.assertTrue(self.tub > self.tlb)
        self.assertTrue(self.oub > self.olb)

        self.assertAlmostEqual(np.round(self.oub, 2), np.round(self.tub.cpu().detach().numpy(), 2))
        self.assertAlmostEqual(np.round(self.olb, 2), np.round(self.tlb.cpu().detach().numpy(), 2))

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
        oub, olb, sim_numpy, smn = original_ublb(dat.batch_size, features=features, uu=uu, ll=ll, train_datasets=train_datasets, index=index)
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

        print(l1)
        print(l2)
        self.assertAlmostEqual(l1.cpu().detach().numpy(), l2.cpu().detach().numpy(), delta=1.)
        # self.assertTrue((g1 == g2).all())

    def test_pseudo_labeling(self):
        feats = torch.Tensor([
            [1., 1., 0., 0.],
            [1., .9, .0, .0],
            [.1, .1, .1, .1],
            [.5, .1, .9, .8],
            [.6, .6, .6, .6],
            [.1, .2, .3, .4],
            [.4, .3, .2, .1],
            [.5, .5, .5, .5],
            [.8, .1, .2, .7],
            [.9, .0, .3, .9],
            ])
        
        index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ions = torch.tensor([0,1,0,3,4,5,6,7,8,9])
        ion_label_mat = pairwise_same_elements(ions)
        
        ub, lb, sim_mat = clust.compute_ublb(feats, 90, 10, None, None)

        # quantiles
        self.assertGreater(ub.cpu().numpy(), 0.99)
        self.assertLess(lb.cpu().numpy(), 0.5)

        self.assertEqual(ion_label_mat.sum(), 12)
        self.assertEqual(ion_label_mat[0, 2], True)

        ion_label_mat = torch.tensor(ion_label_mat.astype(int))

        pos, neg = pseudo_labeling(ub=ub, lb=lb, sim=sim_mat.to(device), 
                                   index=index, knn=False, knn_adj=torch.tensor([]).to(device), 
                                   ion_label_mat=ion_label_mat.to(device), 
                                   dataset_specific_percentiles=False,
                                   dataset_ub=torch.tensor([]), dataset_lb=torch.tensor([]), 
                                   ds_labels=torch.tensor([]), device=device)
        
        self.assertTrue((pos.to(device)[0] == torch.tensor([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.], 
                                                           device=device)).all())
        
        self.assertTrue((neg.to(device)[0] == torch.tensor([0., 0., 0., 1., 0., 1., 0., 0., 0., 1.], 
                                                           device=device)).all())

    def test_dataset_bounds(self):
        feats = torch.Tensor([
            [1., 1., 0., 0.],
            [1., .9, .0, .0],
            [.1, .1, .1, .1],
            [.5, .1, .9, .8],
            [.6, .6, .6, .6],
            [.1, .2, .3, .4],
            [.4, .3, .2, .1],
            [.5, .5, .5, .5],
            [.8, .1, .2, .7],
            [.9, .0, .3, .9],
            ])
        
        ions = torch.tensor([0,1,0,3,4,5,6,7,8,9])

        ds_labels = torch.tensor([0,0,0,1,1,1,0,0,0,0])

        _, _, sim_mat = clust.compute_ublb(feats, 90, 10, None, None)

        dub, dlb = compute_dataset_ublb(sim_mat, ds_labels, 10, 90)

        self.assertGreater(dub[0].cpu().numpy(), 0.99)
        self.assertLess(dlb[0].cpu().numpy(), 0.59)
        
if __name__ == '__main__':
    unittest.main()
