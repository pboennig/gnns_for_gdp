
import torch
import pandas as pd
from torch_geometric.data import Data
import random
import torch.nn.functional as F
import numpy as np
from hyperparams import hyperparams

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
    edge_index = torch.from_numpy(edges[['i_id', 'j_id']].to_numpy(np.long)).t()
    edge_attr = torch.from_numpy(edges[EDGE_FEATURES].to_numpy(np.float32)) #extract the features from the dataset.
    edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))
    
    # load in target values
    y_df = pd.read_csv(f'output/Y_{year}.csv')
    y_df['id'] = y_df['iso_code'].map(iso_code_to_id)
    y = torch.from_numpy(y_df.sort_values('id')[f'{year+1}'].to_numpy(np.float32)).unsqueeze(1)# get labels as tensor
    y = y.log() # log scale since spread of GDP is large
    
    # load in input features
    x_df = pd.read_csv(f'output/X_NODE_{year}.csv')
    x_df['id'] = x_df['iso_code'].map(iso_code_to_id)
    features = ['pop', 'cpi', 'emp']
    x = torch.from_numpy(x_df.sort_values('id').loc[:,features].to_numpy(np.float32))
    x = (x - x.mean(axis=0)) / (x.std(axis=0))  # scale and center data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def evaluate_model(model, data_iter):
    '''
    Accumulate MSE over a data list or loader.
    '''
    return sum([F.mse_loss(model(data), data.y).item() for data in data_iter])

def get_data():
    '''
    Generate data_lists for train, val, and test. These lists can be either loaded into data_loaders
    or indexed directly.
    '''

    data_list = [create_data(year) for year in range(FIRST_YEAR, LAST_YEAR)]
    random.shuffle(data_list)
    data_train = data_list[:NUM_TRAIN]
    data_val = data_list[NUM_TRAIN:NUM_TRAIN+NUM_VAL+1]
    data_test = data_list[NUM_TRAIN+NUM_VAL:]
    return (data_train, data_val, data_test)


def save_gt_vs_prediction(model, data_iter, fname):
    '''
    Given a model and a dataset, save a csv to fname that stores the ground truth
    value and the predicted values in two columns. Used for comparing
    '''
    ground_truth = torch.cat([data.y for data in data_iter], axis=0)
    preds = torch.cat([model(data) for data in data_iter], axis=0) 
    prediction_df = pd.DataFrame({"ground_truth": ground_truth.detach().numpy()[:,0], "prediction": preds.detach().numpy()[:,0]})
    prediction_df.to_csv(fname)

def preds_file(model_type, epoch):
    return f"results/preds/{model_type}_{hyperparams['learning_rate']}_{epoch}_out_of_{hyperparams['n_epochs']}.csv" 

def preds_plot_file(model_type, epoch):
    return f"plots/preds/{model_type}_{hyperparams['learning_rate']}_{epoch}.png"

def loss_file(model_type):
    return f"results/losses/{model_type}_{hyperparams['learning_rate']}_{hyperparams['n_epochs']}.csv"
