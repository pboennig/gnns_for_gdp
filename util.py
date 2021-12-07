
import torch
import pandas as pd
from torch_geometric.data import Data
import random
import torch.nn.functional as F

FIRST_YEAR = 1995
LAST_YEAR = 2019
FEATURES = ['pop', 'cpi', 'emp']
NUM_TRAIN = 15
NUM_VAL = 3
NUM_TEST = 6
NUM_EDGE_FEATURES = 10
EDGE_FEATURES = ['f'+str(i) for i in range(NUM_EDGE_FEATURES)]


def create_data(year):
    '''
    For given year, pull in node features, edge features, and edge index and 
    save in a PyG Data object.
    '''
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
    
    # if year == 1995:
    #     print('edges')
    #     print(edge_attr)
    #     print('wut')
    # return

    # load in target values
    y_df = pd.read_csv(f'output/Y_{year}.csv')
    y_df['id'] = y_df['iso_code'].map(iso_code_to_id)
    y = torch.from_numpy(y_df.sort_values('id')[f'{year+1}'].to_numpy()).unsqueeze(1) # get labels as tensor
    
    # load in input features
    x_df = pd.read_csv(f'output/X_NODE_{year}.csv')
    x_df['id'] = x_df['iso_code'].map(iso_code_to_id)
    features = ['pop', 'cpi', 'emp']
    x_df.loc[:,features] = (x_df.loc[:,features] - x_df.loc[:,features].mean()) / (x_df.loc[:,features].std())
    x = torch.from_numpy(x_df.sort_values('id').loc[:,features].to_numpy())
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def evaluate_model(model, data_val):
    loss = 0.0
    for data in data_val:
        loss += F.mse_loss(model(data), data.y)
    return loss.item()

def get_data():
    data_list = [create_data(year) for year in range(FIRST_YEAR, LAST_YEAR)]
    random.shuffle(data_list)
    data_train = data_list[:NUM_TRAIN]
    data_val = data_list[NUM_TRAIN:NUM_TRAIN+NUM_VAL+1]
    data_test = data_list[NUM_TRAIN+NUM_VAL:]
    return (data_train, data_val, data_test)