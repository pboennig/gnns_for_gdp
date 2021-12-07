import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from util import *

# define model
class GDPModel(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.conv1 = GATConv(self.num_features, self.hidden_size, edge_dim = NUM_EDGE_FEATURES)
        self.conv2 = GATConv(self.hidden_size, self.hidden_size, edge_dim = NUM_EDGE_FEATURES)
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # TODO: figure out if L2 normalizing edge features improve model.
        # edge_attr = torch.nn.functional.normalize(edge_attr, p=2.0, dim=0, eps=1e-12, out=None)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)

        return self.linear(x)