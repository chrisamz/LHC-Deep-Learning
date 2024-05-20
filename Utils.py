import numpy as np
import torch
from torch_geometric.data import Data

def simulate_real_time_data(batch_size, node_feature_size, edge_feature_size):
    while True:
        num_nodes = np.random.randint(20, 100)
        num_edges = np.random.randint(50, 200)
        
        node_features = torch.rand((num_nodes, node_feature_size))
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.rand((num_edges, edge_feature_size))
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        yield data
