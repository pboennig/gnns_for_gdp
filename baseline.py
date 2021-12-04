import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

import csv
import numpy as np
import matplotlib.pyplot as plt

FIRST_YEAR = 1995
LAST_YEAR = 2019
FEATURES = ['pop', 'cpi', 'emp']

def create_data(year):
    assert(year in range(FIRST_YEAR, LAST_YEAR + 1))
    edges = pd.read_csv(f'output/X_EDGE_{year}.csv')

    # generate map from iso_code to ids of form [0, ..., num_unique_iso_codes - 1]
    iso_codes = set(edges['i'])
    iso_codes = iso_codes.union(set(edges['j']))
    iso_code_to_id = {code : i for (i, code) in enumerate(iso_codes)}

    # load in edge index
    edges['i_id'] = edges['i'].map(iso_code_to_id)
    edges['j_id'] = edges['j'].map(iso_code_to_id)
    edge_index = torch.from_numpy(edges[['i_id', 'j_id']].to_numpy()).t()
    
    # load in target values
    y_df = pd.read_csv(f'output/Y_{year}.csv')
    y_df['id'] = y_df['iso_code'].map(iso_code_to_id)
    y = torch.from_numpy(y_df.sort_values('id')[f'{year+1}'].to_numpy()).unsqueeze(1) # get labels as tensor
    
    # load in input features
    x_df = pd.read_csv(f'output/X_NODE_{year}.csv')
    x_df['id'] = x_df['iso_code'].map(iso_code_to_id)
    features = ['pop', 'cpi', 'emp']
    x = torch.from_numpy(x_df.sort_values('id').loc[:,features].to_numpy())
    return Data(x=x, edge_index=edge_index, y=y)

data_list = [create_data(year) for year in range(FIRST_YEAR, LAST_YEAR)]

# define model
class GCN(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=16, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.conv1 = GCNConv(self.num_features, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)

        return self.linear(x)

# train model
model = GCN().double() # needs to be double precision
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loader = DataLoader(data_list, batch_size=2, shuffle=True)

model.train()
losses = []
save_interval = 10
print_interval = 100
for epoch in range(2000):
    if epoch % print_interval == 0:
        print(f"{round((epoch + 1)/2000 * 100, 2)}%", end='\r')
        
    optimizer.zero_grad()
    data = next(iter(loader))
    out = model(data)
    loss = F.mse_loss(out, data.y)
    if epoch % save_interval == 0:
        losses.append(loss.item())
    loss.backward()
    optimizer.step()

# write model results to disk
with open('results/baseline_train.csv', 'w+') as f:
    writer = csv.writer(f)
    for (i, loss) in enumerate(losses):
        writer.writerow([i * save_interval, loss])

# plot results
plt.figure()
plt.plot(losses)
plt.title("GNN Loss")
plt.xlabel("Epoch (hundreds)")
plt.show()
