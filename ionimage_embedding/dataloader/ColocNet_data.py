import torch
from torch_geometric.data import Data


def get_data(
        node_features: torch.Tensor,
        pos_edge_list: torch.Tensor,
        neg_edge_list: torch.Tensor,
        edge_weights: torch.Tensor,
             ) -> Data:
    
    return Data(x=node_features,
                edge_index=pos_edge_list,
                neg_edge_index=neg_edge_list,
                edge_attr=edge_weights)

# Creats a torch_geometric.data.Data object for each coloc graph
# Creates positive and negative edge list that is also used for evaluation
# Object holds information about topN cutoffs and feature size that is curcial
# for the initializing the model
# Holds the dataset object and function to return the dataloader