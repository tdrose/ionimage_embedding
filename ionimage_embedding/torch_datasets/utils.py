from typing import Optional
import numpy as np

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
