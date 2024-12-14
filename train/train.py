import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
import numpy as np

class EmotionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionGNN, self).__init__()

        # Graph CNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fully Connected Layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global Pooling (mean)
        x = torch.mean(x, dim=0)

        # Fully Connected Layer for Classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)