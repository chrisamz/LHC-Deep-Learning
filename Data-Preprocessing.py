import numpy as np
import torch
from torch_geometric.data import Data
from utils import simulate_real_time_data

def preprocess_data(batch_size, node_feature_size, edge_feature_size):
    data_stream = simulate_real_time_data(batch_size, node_feature_size, edge_feature_size)
    dataset = []
    for _ in range(batch_size):
        data = next(data_stream)
        dataset.append(data)
    return dataset

if __name__ == "__main__":
    batch_size = 10
    node_feature_size = 30
    edge_feature_size = 14
    dataset = preprocess_data(batch_size, node_feature_size, edge_feature_size)
    print("Data preprocessing completed.")
