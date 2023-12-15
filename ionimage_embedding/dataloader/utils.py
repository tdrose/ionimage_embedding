from metaspace import SMInstance
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from typing import Dict, Tuple

def size_adaption(image_dict: Dict[str, np.ndarray]):
    maxh = np.max(np.array([x.shape[1] for x in image_dict.values()]))
    maxw = np.max(np.array([x.shape[2] for x in image_dict.values()]))
    
    out_dict = {}
    for dsid, imgs in image_dict.items():
        # Height
        if imgs.shape[1] == maxh:
            pad1: Tuple[int, int] = (0, 0)
        else:
            hdiff = maxh - imgs.shape[1]
            pad1: Tuple[int, int] = (hdiff//2, hdiff//2 + hdiff%2)
        
        # Width
        if imgs.shape[2] == maxw:
            pad2: Tuple[int, int] = (0, 0)
        else:
            wdiff = maxw - imgs.shape[2]
            pad2: Tuple[int, int] = (wdiff//2, wdiff//2 + wdiff%2)

        out_dict[dsid] = np.pad(imgs, ((0, 0), pad1, pad2), 
                                constant_values=0) # type: ignore
    
    return out_dict


def size_adaption_symmetric(image_dict: Dict[str, np.ndarray]):
    maxh = np.max(np.array([x.shape[1] for x in image_dict.values()]))
    maxw = np.max(np.array([x.shape[2] for x in image_dict.values()]))

    absmax = max(maxh, maxw)

    out_dict = {}
    for dsid, imgs in image_dict.items():
        # Height
        if imgs.shape[1] == absmax:
            pad1: Tuple[int, int] = (0, 0)
        else:
            hdiff = absmax - imgs.shape[1]
            pad1: Tuple[int, int] = (hdiff // 2, hdiff // 2 + hdiff % 2)

        # Width
        if imgs.shape[2] == absmax:
            pad2: Tuple[int, int] = (0, 0)
        else:
            wdiff = absmax - imgs.shape[2]
            pad2: Tuple[int, int] = (wdiff // 2, wdiff // 2 + wdiff % 2)

        out_dict[dsid] = np.pad(imgs, ((0, 0), pad1, pad2), constant_values=0) # type: ignore

    return out_dict

def download_data(ds_ids, db=("HMDB", "v4"), fdr=0.2, scale_intensity='TIC', 
                  colocml_preprocessing=False, maxzero=.95):
    
    sm = SMInstance()
    
    training_results = {}
    training_images = {}
    training_if = {}
    
    for k in ds_ids:
        ds = sm.dataset(id=k)
        
        results = ds.results(database=db, fdr=fdr).reset_index()
        training_results[k] = results
        tmp = ds.all_annotation_images(fdr=fdr, database=db, only_first_isotope=True, 
                                       scale_intensity=scale_intensity, hotspot_clipping=False)
        
        onsample = dict(zip(results['formula'].str.cat(results['adduct']), ~results['offSample']))
        formula = [x.formula+x.adduct for x in tmp if onsample[x.formula+x.adduct]]
        tmp = np.array([x._images[0] for x in tmp if onsample[x.formula+x.adduct]])

        if colocml_preprocessing:
            a = ndimage.median_filter(tmp, size=(1,3,3))
            b: np.ndarray = a.reshape((a.shape[0], -1))
            mask = b < np.percentile(b, q=50, axis=1)[:, np.newaxis]
            b[mask] = 0.
            tmp = b.reshape(a.shape)

        training_images[k] = tmp
        training_if[k] = formula
    
    # Transform all images to squares of the same size
    padding_images = size_adaption_symmetric(training_images)
    
    training_data = []
    training_datasets = [] 
    training_ions = []
    
    for dsid, imgs in padding_images.items():

        training_data.append(imgs)
        training_datasets += [dsid] * imgs.shape[0]
        training_ions += training_if[dsid]
        
    training_data = np.concatenate(training_data)
    training_datasets = np.array(training_datasets)
    training_ions = np.array(training_ions)

    # Filter out images that are mainly zero
    imgsize = training_data.shape[1]*training_data.shape[2]
    zero_count = np.sum((training_data == 0).reshape((training_data.shape[0], -1)), axis=1)
    zero_mask = zero_count >= (imgsize*maxzero)

    # Old version: filter out only all zero images
    # zero_mask = (training_data == 0).reshape((training_data.shape[0], -1)).all(axis=1)

    return training_data[~zero_mask], training_datasets[~zero_mask], training_ions[~zero_mask]

def pairwise_same_elements(int_list):
    # Convert the list to a NumPy array
    int_array = np.array(int_list)
    
    # Create a pairwise comparison matrix
    pairwise_matrix = int_array[:, None] == int_array
    
    return pairwise_matrix

def make_symmetric(matrix):
    # Use the maximum of the matrix and its transpose to make it symmetric
    symmetric_matrix = np.maximum(matrix, matrix.T)

    # Return the resulting symmetric matrix
    return symmetric_matrix

def run_knn(features: np.ndarray, k: int = 10):
    # Todo: Make a better solution for excluding self neighborhood
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(features)
    # TODO: This takes way too long... Implement faster version
    _, indices = nbrs.kneighbors(features)
    # Excluding self neighborhood here
    idx = indices[:, 1:]
    adj = np.zeros((features.shape[0], features.shape[0]))
    for i in range(len(idx)):
        adj[i, idx[i]] = 1
    return make_symmetric(adj)