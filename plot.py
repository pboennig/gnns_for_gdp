import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util import *

sns.set_theme()

def loss_plot(prefix):
    loss_df = pd.read_csv(f'results/{prefix}_train.csv', index_col=0)
    loss_df = loss_df.melt('epoch', var_name='loss_type', value_name='loss')
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', hue='loss_type', kind='line').set(title=f"{prefix} training trajectory")
    loss_plot.fig.savefig(f'plots/{prefix}_loss.png', dpi=400)
    plt.close(plt.gcf())

def pred_plot(input_csv, title, out_file, bounding_scaling_factor=.15):
    preds_df = pd.read_csv(input_csv)
    # make limits nicely surround points
    min_val = preds_df[['ground_truth','prediction']].min().values.min() * (1 - bounding_scaling_factor)
    max_val = preds_df[['ground_truth', 'prediction']].max().values.max() * (1 + bounding_scaling_factor)
    lim = (min_val, max_val)
    preds_plot = sns.relplot(data=preds_df, x='ground_truth', y='prediction')
    preds_plot.set(xscale='log', yscale='log') # large spread in values
    preds_plot.set(xlabel='actual GDP', ylabel='predicted GDP') # label axes
    preds_plot.set(xlim=lim, ylim=lim) # limits must be same to match intution that y=x is correct
    preds_plot.set(title=title)
    preds_plot.fig.savefig(out_file, dpi=400)
    plt.close(plt.gcf())

def hyperparams_plot(model_type):
    sweep_df = pd.read_csv(f'results/{model_type}_hyperparams.csv', index_col=0)
    sweep_plot = sns.relplot(data=sweep_df, x='lr', y='val_loss')
    sweep_plot.set(title=f"{model_type} hyperparameter sweep")
    sweep_plot.set(ylim=(1e23, 1e25))
    sweep_plot.fig.savefig(f'plots/{model_type}_sweep.png', dpi=400)
    plt.close(plt.gcf())


for model_type in ['baseline', 'model']:
    #hyperparams_plot(model_type)
    for lr in get_sweep_range():
        #loss_plot(f"{model_type}_{lr}_100")
        for e in range(0, 1001, 100):
            pred_plot(f"results/{model_type}_{lr}_{e}_out_of_1000_prediction.csv", f"Prediction after {e} epochs", f"plots/{model_type}_{lr}_{e}.png")

