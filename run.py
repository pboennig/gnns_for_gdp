import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from util import *
from baseline import BaselineGDPModel
from model import GDPModel


#
data_train, data_val, data_test = get_data()

# First, train baseline.
# train model
# Hyperparameters
batch_size = 3
n_epochs = 200
save_loss_interval = 10
print_interval = 50 
save_model_interval = 50

def train(name_prefix, model, batch_size, learning_rate, n_epochs, save_loss_interval, print_interval, save_model_interval):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            val_loss = evaluate_model(model, data_val) 
            if epoch % print_interval == 0:
                print(f"Epoch: {epoch} Train loss: {epoch_loss / NUM_TRAIN} Validation loss: {evaluate_model(model, data_val) / NUM_VAL}")
            losses.append((epoch, epoch_loss / NUM_TRAIN, evaluate_model(model, data_val)/ NUM_VAL))
        if epoch % save_model_interval == 0:
            model.eval()
            prediction_df = pd.DataFrame({'ground_truth': test_data.y.detach().numpy()[:,0], 'prediction': model(test_data).detach().numpy()[:,0]})
            prediction_df.to_csv(f"results/{name_prefix}_{learning_rate}_{epoch}_out_of_{n_epochs}_prediction.csv")
            torch.save(model.state_dict(), f"models/{name_prefix}_{learning_rate}_{epoch}_out_of_{n_epochs}.pt")


    loss_df = pd.DataFrame(losses, columns=['epoch', 'train', 'val'])
    loss_df.to_csv(f"results/{name_prefix}_{learning_rate}_{n_epochs}_train.csv")
    return evaluate_model(model, data_val)


baseline_loss = []
model_loss = []
for lr in get_sweep_range():
    model = BaselineGDPModel().double() # needs to be double precision
    baseline_val_loss = train("baseline", model, batch_size, lr, n_epochs, save_loss_interval, print_interval, save_model_interval)
    baseline_loss.append((lr, baseline_val_loss))
    model = GDPModel().double() # needs to be double precision
    model_val_loss = train("model", model, batch_size, lr, n_epochs, save_loss_interval, print_interval, save_model_interval)
    model_loss.append((lr, model_val_loss))
    print(lr, baseline_val_loss, model_val_loss)

pd.DataFrame(baseline_loss, columns=['lr', 'val_loss']).to_csv('results/baseline_hyperparams.csv')
pd.DataFrame(model_loss, columns=['lr', 'val_loss']).to_csv('results/model_hyperparams.csv')