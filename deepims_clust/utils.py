import numpy as np


# Define a function to make a symmetric matrix from a non-symmetric matrix
def make_symmetric(matrix):
    # Use the maximum of the matrix and its transpose to make it symmetric
    symmetric_matrix = np.maximum(matrix, matrix.T)

    # Return the resulting symmetric matrix
    return symmetric_matrix

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