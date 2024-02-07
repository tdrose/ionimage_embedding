from metaspace import SMInstance
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from typing import Dict, Tuple, List, Optional
import os
import uuid
import pickle

from .constants import ION_IMAGE_DATA, DATASET_DATA, COLOC_NET_DISCRETE_DATA


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


def size_adaption_symmetric(image_dict: Dict[str, np.ndarray], 
                            vitb16_compatible: bool=False, 
                            force_size: Optional[int]=None):
    maxh = np.max(np.array([x.shape[1] for x in image_dict.values()]))
    maxw = np.max(np.array([x.shape[2] for x in image_dict.values()]))

    absmax = max(maxh, maxw)
    if force_size is not None:
        if force_size >= absmax:
            absmax = force_size
        else:
            raise ValueError('force_size must be larger than the largest image dimension.')
    elif vitb16_compatible:
        # Check if divisible by 16
        if absmax % 16 != 0:
            absmax = absmax + (16 - (absmax % 16))

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

def download_data(ds_ids: List[str], db: Tuple[str, str]=("HMDB", "v4"), fdr: float=0.2, 
                  scale_intensity: str='TIC', 
                  colocml_preprocessing: bool=False, maxzero: float=.95, 
                  vitb16_compatible: bool=False, 
                  force_size: Optional[int]=None,
                  min_images: int=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
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
    padding_images = size_adaption_symmetric(training_images, 
                                             vitb16_compatible=vitb16_compatible,
                                             force_size=force_size)
    
    training_data = []
    training_datasets = [] 
    training_ions = []
    
    for dsid, imgs in padding_images.items():
        if imgs.shape[0] >= min_images:
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

    training_data = training_data[~zero_mask]
    training_datasets = training_datasets[~zero_mask]
    training_ions = training_ions[~zero_mask]

    # Remove images from datasets with less than min_images
    dataset_counts = np.unique(training_datasets, return_counts=True)
    mask = np.isin(training_datasets, dataset_counts[0][dataset_counts[1] >= min_images])
    training_data = training_data[mask]
    training_datasets = training_datasets[mask]
    training_ions = training_ions[mask]

    return training_data, training_datasets, training_ions

def cache_hashing(dataset_ids: List[str], colocml_preprocessing: bool,
                  db: Tuple[str, str], fdr: float, scale_intensity: str,
                  force_size: Optional[int], maxzero: float, vitb16_compatible: bool, 
                  min_images: int) -> str:
    cache_hex = '{}_{}_{}_{}-{}_{}_{}_{}_{}_{}_{}'.format(ION_IMAGE_DATA,
                                                     ''.join(dataset_ids), 
                                                     colocml_preprocessing, 
                                                     str(db[0]), str(db[1]), 
                                                     str(fdr), str(scale_intensity),
                                                     str(maxzero), str(vitb16_compatible),
                                                     str(force_size), str(min_images)
                                                    )
    
    return uuid.uuid5(uuid.NAMESPACE_URL, cache_hex).hex


def get_data(dataset_ids: List[str], cache: bool=True, cache_folder: str='cache', 
             db: Tuple[str, str]=('HMDB', 'v4'), 
             fdr: float=0.2, scale_intensity: str='TIC', 
             colocml_preprocessing: bool=False, force_size: Optional[int]=None,
             min_images: int=5,
             maxzero: float=.95, vitb16_compatible: bool=False) -> Tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray]:
    if cache:
        # make hash of datasets
        cache_hex = cache_hashing(dataset_ids, colocml_preprocessing, db, fdr, scale_intensity,
                                  force_size, maxzero, vitb16_compatible, min_images)
        
        cache_file = '{}_{}.pickle'.format(ION_IMAGE_DATA, cache_hex)

        # Check if cache folder exists
        if not os.path.isdir(cache_folder):
            os.mkdir(cache_folder)

        # Download data if it does not exist
        if cache_file not in os.listdir(cache_folder):
            tmp = download_data(dataset_ids, db=db, 
                                fdr=fdr, 
                                scale_intensity=scale_intensity, 
                                colocml_preprocessing=colocml_preprocessing, 
                                maxzero=maxzero, vitb16_compatible=vitb16_compatible, 
                                force_size=force_size, min_images=min_images)
            data, dataset_labels, ion_labels = tmp

            pickle.dump((data, dataset_labels, ion_labels), 
                        open(os.path.join(cache_folder, cache_file), "wb"))
            print('Saved file: {}'.format(os.path.join(cache_folder, cache_file)))      
        
        # Load cached data
        else:
            print('Loading cached data from: {}'.format(os.path.join(cache_folder, cache_file)))
            tmp = pickle.load(open(os.path.join(cache_folder, cache_file), "rb" ) )
            data, dataset_labels, ion_labels = tmp
    else:
        tmp = download_data(dataset_ids, db=db, fdr=fdr, scale_intensity=scale_intensity, 
                            colocml_preprocessing=colocml_preprocessing, 
                            maxzero=maxzero, vitb16_compatible=vitb16_compatible, 
                            force_size=force_size, min_images=min_images)
        data, dataset_labels, ion_labels = tmp

    return data, dataset_labels, ion_labels


def purge_cache(cache_folder: str='cache'):
    purged_files = []
    for f in os.listdir(cache_folder):
        if f.startswith(ION_IMAGE_DATA):
            purged_files.append(f)
            os.remove(os.path.join(cache_folder, f))
        elif f.startswith(DATASET_DATA):
            purged_files.append(f)
            os.remove(os.path.join(cache_folder, f))
        elif f.startswith(COLOC_NET_DISCRETE_DATA):
            purged_files.append(f)
            # Data is stored in a folder
            os.rmdir(os.path.join(cache_folder, f))
        else:
            pass
    print('Deleted files:')
    for i in purged_files:
        print('* ', i)


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


def create_edgelist(edges: Optional[np.ndarray], i: int, top_k: int,
                                top_k_idx: np.ndarray) -> np.ndarray:
                
    new_edges = np.stack([np.repeat(i, top_k), 
                          top_k_idx], axis=0).transpose()

    # Make undirected
    new_edges = np.vstack([new_edges, new_edges[:, ::-1]])

    if edges is None:
        return np.unique(new_edges, axis=0)
    else:
        tmp = np.vstack([new_edges, edges])
        return np.unique(tmp, axis=0)
