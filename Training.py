import torch
from torch import nn
from torch_geometric.data import DataLoader
from model import InteractionNetwork
from data_preprocessing import preprocess_data

node_feature_size = 30
edge_feature_size = 14
batch_size = 10
learning_rate = 0.001
epochs = 10

model = InteractionNetwork(node_feature_size, edge_feature_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

dataset = preprocess_data(batch_size, node_feature_size, edge_feature_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(model, loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            target = torch.rand((data.x.size(0), 1))  # Dummy target
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader)}')

if __name__ == "__main__":
    train(model, loader, optimizer, criterion, epochs)
    torch.save(model.state_dict(), 'interaction_network.pth')
    print("Training completed and model saved.")
