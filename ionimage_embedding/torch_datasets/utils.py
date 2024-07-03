from typing import Optional
import numpy as np

def create_edgelist(edges: Optional[np.ndarray], i: int, top_k: int,
                    top_k_idx: np.ndarray) -> np.ndarray:
    """
    Create or extend an undirected edgelist.

    Args:
        edges: Optional. Existing edgelist, if available, new edge will be added to it
        i: Index of the new source node
        top_k: Number of edges to add
        top_k_idx: Indices of the nodes to connect node i to.
    """
                
    new_edges = np.stack([np.repeat(i, top_k), 
                          top_k_idx], axis=0).transpose()

    # Make undirected
    new_edges = np.vstack([new_edges, new_edges[:, ::-1]])

    if edges is None:
        return np.unique(new_edges, axis=0)
    else:
        tmp = np.vstack([new_edges, edges])
        return np.unique(tmp, axis=0)
