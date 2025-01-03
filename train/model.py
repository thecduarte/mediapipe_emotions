import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First graph convolution
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second graph convolution
        self.fc = nn.Linear(hidden_dim, label_dim)  # Fully connected layer for classification

    def forward(self, data):
        # Data has 'x' as the node features and 'edge_index' as the graph structure
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolution layers with ReLU activations
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling to aggregate node features into a graph-level representation
        x = global_mean_pool(x, data.batch)  # data.batch maps nodes to graphs
        
        # Final classification layer
        x = self.fc(x)
        return x