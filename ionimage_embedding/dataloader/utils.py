from metaspace import SMInstance
import numpy as np
from sklearn.neighbors import NearestNeighbors

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


def size_adaption_symmetric(image_dict: dict):
    maxh = np.max([x.shape[1] for x in image_dict.values()])
    maxw = np.max([x.shape[2] for x in image_dict.values()])

    absmax = max(maxh, maxw)

    out_dict = {}
    for dsid, imgs in image_dict.items():
        # Height
        if imgs.shape[1] == absmax:
            pad1 = (0, 0)
        else:
            hdiff = absmax - imgs.shape[1]
            pad1 = (hdiff // 2, hdiff // 2 + hdiff % 2)

        # Width
        if imgs.shape[2] == absmax:
            pad2 = (0, 0)
        else:
            wdiff = absmax - imgs.shape[2]
            pad2 = (wdiff // 2, wdiff // 2 + wdiff % 2)

        out_dict[dsid] = np.pad(imgs, ((0, 0), pad1, pad2), constant_values=0)

    return out_dict

def download_data(ds_ids, db=("HMDB", "v4"), fdr=0.2, scale_intensity='TIC', hotspot_clipping=False):
    
    sm = SMInstance()
    
    training_results = {}
    training_images = {}
    training_if = {}
    
    for k in ds_ids:
        ds = sm.dataset(id=k)
        
        results = ds.results(database=db, fdr=fdr).reset_index()
        training_results[k] = results
        tmp = ds.all_annotation_images(fdr=fdr, database=db, only_first_isotope=True, 
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

        training_data.append(imgs)
        training_datasets += [dsid] * imgs.shape[0]
        training_ions += training_if[dsid]
        
    training_data = np.concatenate(training_data)
    training_datasets = np.array(training_datasets)
    training_ions = np.array(training_ions)

    return training_data, training_datasets, training_ions

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