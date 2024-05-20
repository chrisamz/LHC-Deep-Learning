pip install torch torch-geometric numpy

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class InteractionNetwork(MessagePassing):
    def __init__(self, node_feature_size, edge_feature_size):
        super(InteractionNetwork, self).__init__(aggr='add')
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.mlp1 = Seq(Lin(2 * node_feature_size + edge_feature_size, 128), ReLU(), Lin(128, node_feature_size))
        self.mlp2 = Seq(Lin(node_feature_size, 64), ReLU(), Lin(64, 1))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp1(tmp)

    def update(self, aggr_out):
        return self.mlp2(aggr_out)

# Example node and edge feature sizes (adjust as needed)
node_feature_size = 30
edge_feature_size = 14

model = InteractionNetwork(node_feature_size, edge_feature_size)

import numpy as np

def simulate_real_time_data(batch_size, node_feature_size, edge_feature_size):
    while True:
        num_nodes = np.random.randint(20, 100)
        num_edges = np.random.randint(50, 200)
        
        node_features = torch.rand((num_nodes, node_feature_size))
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = torch.rand((num_edges, edge_feature_size))
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        yield data

# Simulate real-time data stream
data_stream = simulate_real_time_data(batch_size=1, node_feature_size=node_feature_size, edge_feature_size=edge_feature_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def process_real_time_data(model, data_stream, optimizer, criterion):
    model.train()
    for data in data_stream:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        # Dummy target for illustration purposes
        target = torch.rand((data.x.size(0), 1))
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f'Processed a batch with loss: {loss.item()}')

# Start processing the real-time data stream
process_real_time_data(model, data_stream, optimizer, criterion)

