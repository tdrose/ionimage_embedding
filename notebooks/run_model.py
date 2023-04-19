from metaspace import SMInstance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepims_clust import DeepClustering

sm = SMInstance()

kidney_datasets = [
    '2022-12-07_02h13m50s',
    '2022-12-07_02h13m20s',
    '2022-12-07_02h10m45s',
    '2022-12-07_02h09m41s',
    '2022-12-07_02h08m52s'
                  ]


kidney_results = {}
kidney_images = {}
kidney_if = {}
polarity = '+'

for k in kidney_datasets:
    ds = sm.dataset(id=k)
    results = ds.results(database=("HMDB", "v4"), fdr=0.2).reset_index()
    kidney_results[k] = results
    tmp = ds.all_annotation_images(fdr=0.2, database=("HMDB", "v4"), only_first_isotope=True)
    onsample = dict(zip(results['formula'].str.cat(results['adduct']), ~results['offSample']))
    formula = [x.formula+x.adduct for x in tmp if onsample[x.formula+x.adduct]]
    tmp = np.array([x._images[0] for x in tmp if onsample[x.formula+x.adduct]])
    kidney_images[k] = tmp
    kidney_if[k] = formula
    
    
def size_adaption(image_dict: dict):
    maxh = np.max([x.shape[1] for x in image_dict.values()])
    maxw = np.max([x.shape[2] for x in image_dict.values()])
    
    out_dict = {}
    for dsid, imgs in image_dict.items():
        # Height
        if imgs.shape[1] == maxh:
            pad1 = (0, 0)
        else:
            hdiff = maxh - imgs.shape[1]
            pad1 = (hdiff//2, hdiff//2 + hdiff%2)
        
        # Width
        if imgs.shape[2] == maxw:
            pad2 = (0, 0)
        else:
            wdiff = maxw - imgs.shape[2]
            pad2 = (wdiff//2, wdiff//2 + wdiff%2)

        out_dict[dsid] = np.pad(imgs, ((0, 0), pad1, pad2), constant_values=0)
    
    return out_dict



training_data = []
training_datasets = [] 
training_ions = []

kidney_images_pad = size_adaption(kidney_images)

for dsid, imgs in kidney_images_pad.items():
    
    training_data.append(imgs)
    training_datasets += [dsid] * imgs.shape[0]
    training_ions += kidney_if[dsid]
    
training_data = np.concatenate(training_data)
training_datasets = np.array(training_datasets)
training_ions = np.array(training_ions)



model = DeepClustering(
                images=training_data,
                dataset_labels=training_datasets,
                ion_labels=training_ions,
                num_cluster=7,
                initial_upper=0.99,
                initial_lower=0.01,
                upper_iteration=2,
                lower_iteration=2,
                knn=False, k=10,
                lr=0.0001, batch_size=100,
                pretraining_epochs= 11,
                training_epochs=11,
                cae_encoder_dim=14,
                use_gpu=True
            )


cae, clust = model.train()



