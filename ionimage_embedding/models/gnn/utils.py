from typing import Tuple

import torch

def compute_union_graph(graphs) -> Tuple[torch.Tensor, torch.Tensor]:

    new_edges = []
    nodes = []
    for graph in graphs:
        new_edges.append(torch.stack([graph.x[graph.edge_index[0]], graph.x[graph.edge_index[1]]]))
        nodes.append(graph.x)
    
    new_edges = torch.cat(new_edges, dim=1)
    new_edges = torch.unique(new_edges.transpose(0, 1), dim=0)
    
    nodes = torch.cat(nodes, dim=0)
    nodes = torch.unique(nodes)

    # Map edges to new node index
    for i in range(new_edges.shape[0]):
        new_edges[i, 0] = torch.where(nodes == new_edges[i, 0])[0][0]
        new_edges[i, 1] = torch.where(nodes == new_edges[i, 1])[0][0]


    # Create new edge index
    return nodes, new_edges.transpose(0,1)
