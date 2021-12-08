import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from util import *
from baseline import BaselineGDPModel
from model import GDPModel
from enum import Enum
from hyperparams import hyperparams

data_train, data_val, data_test = get_data()


def train(name_prefix, hyperparams):
    ''' 
    Train model with given hyperparams dict.

    Saves the following CSVs over the course of training:
    1. the loss trajectory: the val and train loss every save_loss_interval epochs at
       filename 'results/{name_prefix}_{learning_rate}_train.csv' e.g. 'results/baseline_0.05_train.csv'
    2. every save_model_interval save both the model at e.g. 'models/baseline_0.05_0_out_of_1000.pt`
       and the predicted values vs actual values in `results/baseline_0.05_0_out_of_1000_prediction.csv' on the test data.
    '''
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']
    save_model_interval = hyperparams['save_model_interval']

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    losses = []
    test_data = data_test[0]
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
        if epoch % save_loss_interval == 0:
            val_loss = evaluate_model(model, data_val) / NUM_VAL
            train_loss = epoch_loss / NUM_TRAIN
            if epoch % print_interval == 0:
                print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}".format(epoch, train_loss, val_loss))
            losses.append((epoch, epoch_loss / NUM_TRAIN, evaluate_model(model, data_val)/ NUM_VAL))
        if epoch % save_model_interval == 0:
            # save predictions for plotting
            model.eval()
            prediction_df = pd.DataFrame({'ground_truth': test_data.y.detach().numpy()[:,0], 'prediction': model(test_data).detach().numpy()[:,0]})
            prediction_df.to_csv(f"results/{name_prefix}_{learning_rate}_{epoch}_out_of_{n_epochs}_prediction.csv")
            save_gt_vs_prediction(model, data_train, f"results/preds/{name_prefix}_{learning_rate}_out_of_{n_epochs}.csv" )
            torch.save(model.state_dict(), f"models/{name_prefix}_{learning_rate}_{epoch}_out_of_{n_epochs}.pt")

    prediction_df = pd.DataFrame({'ground_truth': test_data.y.detach().numpy()[:,0], 'prediction': model(test_data).detach().numpy()[:,0]})
    prediction_df.to_csv(f"results/{name_prefix}_{learning_rate}_prediction.csv")
    torch.save(model.state_dict(), f"models/{name_prefix}_{learning_rate}.pt")
    loss_df = pd.DataFrame(losses, columns=['epoch', 'train', 'val'])
    loss_df.to_csv(f"results/{name_prefix}_{learning_rate}_{n_epochs}_train.csv")
    return evaluate_model(model, data_val)


print("Training baseline...")
model = BaselineGDPModel()
baseline_val_loss = train("baseline", hyperparams)

print("Training model...")
model = GDPModel() # needs to be double precision
model_val_loss = train("model", hyperparams)

print('Baseline validation MSE:', baseline_val_loss)
print('Model validation MSE:', model_val_loss)