import torch
from torch_geometric.data import DataLoader
from model import InteractionNetwork
from data_preprocessing import preprocess_data

node_feature_size = 30
edge_feature_size = 14
batch_size = 10

model = InteractionNetwork(node_feature_size, edge_feature_size)
model.load_state_dict(torch.load('interaction_network.pth'))

dataset = preprocess_data(batch_size, node_feature_size, edge_feature_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def validate(model, loader):
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr)
            target = torch.rand((data.x.size(0), 1))  # Dummy target
            loss = criterion(out, target)
            total_loss += loss.item()
    print(f'Validation Loss: {total_loss/len(loader)}')

if __name__ == "__main__":
    validate(model, loader)
