import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
import random

FIRST_YEAR = 1995
LAST_YEAR = 2019
FEATURES = ['pop', 'cpi', 'emp']
NUM_TRAIN = 15
NUM_VAL = 3
NUM_TEST = 6
NUM_EDGE_FEATURES = 10
EDGE_FEATURES = ['f'+str(i) for i in range(NUM_EDGE_FEATURES)]

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
    edge_attr = torch.from_numpy(edges[EDGE_FEATURES].to_numpy()) #extract the features from the dataset.

    # load in target values
    y_df = pd.read_csv(f'output/Y_{year}.csv')
    y_df['id'] = y_df['iso_code'].map(iso_code_to_id)
    y = torch.from_numpy(y_df.sort_values('id')[f'{year+1}'].to_numpy()).unsqueeze(1) # get labels as tensor
    
    # load in input features
    x_df = pd.read_csv(f'output/X_NODE_{year}.csv')
    x_df['id'] = x_df['iso_code'].map(iso_code_to_id)
    features = ['pop', 'cpi', 'emp']
    x = torch.from_numpy(x_df.sort_values('id').loc[:,features].to_numpy())
    return Data(x=x, edge_index=edge_index,edge_attr=edge_attr y=y)

data_list = [create_data(year) for year in range(FIRST_YEAR, LAST_YEAR)]
random.shuffle(data_list)
data_train = data_list[:NUM_TRAIN]
data_val = data_list[NUM_TRAIN:NUM_TRAIN+NUM_VAL+1]
data_test = data_list[NUM_TRAIN+NUM_VAL:]
# define model
class GDPModel(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.conv1 = GATConv(self.num_features, self.hidden_size)
        self.conv2 = GATConv(self.hidden_size, self.hidden_size)
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

def evaluate_model(model, data_val):
    loss = 0.0
    for data in data_val:
        loss += F.mse_loss(model(data), data.y)
    return loss.item()


# train model
# Hyperparameters
batch_size = 3
learning_rate = 5e-3
n_epochs = 1000 
save_interval = 10
print_interval = 50 


model = GDPModel().double() # needs to be double precision
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

losses = []
for epoch in range(n_epochs):
        
    epoch_loss = 0
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        epoch_loss += loss.item() 
        loss.backward()
        optimizer.step()
    if epoch % save_interval == 0:
        val_loss = evaluate_model(model, data_val) 
        if epoch % print_interval == 0:
            print(f"Epoch: {epoch} Train loss: {epoch_loss / NUM_TRAIN} Validation loss: {evaluate_model(model, data_val) / NUM_VAL}")
        losses.append((epoch, epoch_loss / NUM_TRAIN, evaluate_model(model, data_val)/ NUM_VAL))

loss_df = pd.DataFrame(losses, columns=['epoch', 'train', 'val'])
loss_df.to_csv("results/baseline_train.csv")

model.eval()
test_data = data_test[0]
prediction_df = pd.DataFrame({'ground_truth': test_data.y.detach().numpy()[:,0], 'prediction': model(test_data).detach().numpy()[:,0]})
prediction_df.to_csv("results/prediction.csv")

