from metaspace import SMInstance
import numpy as np
from ..models.clr.utils import size_adaption, size_adaption_symmetric

def download_data(ds_ids, db=("HMDB", "v4"), fdr=0.2, scale_intensity='TIC', hotspot_clipping=False):
    
    sm = SMInstance()
    
    training_results = {}
    training_images = {}
    training_if = {}
    
    for k in ds_ids:
        ds = sm.dataset(id=k)
        
        results = ds.results(database=db, fdr=fdr).reset_index()
        training_results[k] = results
        tmp = ds.all_annotation_images(fdr=fdr, database=fdr, only_first_isotope=True, 
                                       scale_intensity=scale_intensity, hotspot_clipping=hotspot_clipping)
        
        onsample = dict(zip(results['formula'].str.cat(results['adduct']), ~results['offSample']))
        formula = [x.formula+x.adduct for x in tmp if onsample[x.formula+x.adduct]]
        tmp = np.array([x._images[0] for x in tmp if onsample[x.formula+x.adduct]])
        training_images[k] = tmp
        training_if[k] = formula
    
    # Transform all images to squares of the same size
    padding_images = size_adaption_symmetric(training_images)
    
    training_data = []
    training_datasets = [] 
    training_ions = []
    
    for dsid, imgs in padding_images.items():

        if dsid in training_dsid:
            training_data.append(imgs)
            training_datasets += [dsid] * imgs.shape[0]
            training_ions += training_if[dsid]

    return training_data, training_datasets, training_ions