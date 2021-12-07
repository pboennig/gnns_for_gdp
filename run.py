import torch
import pandas as pd
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
learning_rate = 5e-3
n_epochs = 100 
save_interval = 10
print_interval = 50 

print("training baseline...")
model = BaselineGDPModel().double() # needs to be double precision
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
prediction_df.to_csv("results/baseline_prediction.csv")

print("training real model...")
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
loss_df.to_csv("results/model_train.csv")

model.eval()
test_data = data_test[0]
prediction_df = pd.DataFrame({'ground_truth': test_data.y.detach().numpy()[:,0], 'prediction': model(test_data).detach().numpy()[:,0]})
prediction_df.to_csv("results/model_prediction.csv")