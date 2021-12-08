import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util import *
from hyperparams import hyperparams

sns.set_theme()

def loss_plot(prefix):
    loss_df = pd.read_csv(f'results/{prefix}_train.csv', index_col=0)
    loss_df = loss_df.melt('epoch', var_name='loss_type', value_name='loss')
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', hue='loss_type', kind='line').set(title=f"{prefix} training trajectory")
    loss_plot.fig.savefig(f'plots/{prefix}_loss.png', dpi=400)
    plt.close(plt.gcf())

def pred_plot(input_csv, title, out_file, bounding_scaling_factor=.15):
    print(f"Making rediction plot at {out_file}...", end='')
    preds_df = pd.read_csv(input_csv)
    # make limits nicely surround points
    min_val = preds_df[['ground_truth','prediction']].min().values.min() * (1 - bounding_scaling_factor)
    max_val = preds_df[['ground_truth', 'prediction']].max().values.max() * (1 + bounding_scaling_factor)
    lim = (min_val, max_val)
    preds_plot = sns.relplot(data=preds_df, x='ground_truth', y='prediction')
    plt.subplots_adjust(left=0.2) # prevent x-label from getting cut off
    #preds_plot.set(xscale='log', yscale='log') # large spread in values
    preds_plot.set(xlabel='actual GDP', ylabel='predicted GDP') # label axes
    preds_plot.set(xlim=lim, ylim=lim) # limits must be same to match intution that y=x is correct
    preds_plot.set(title=title)
    #plt.tight_layout()
    preds_plot.fig.savefig(out_file, dpi=400)
    plt.close(plt.gcf())
    print("done!")

def hyperparams_plot(model_type):
    sweep_df = pd.read_csv(f'results/{model_type}_hyperparams.csv', index_col=0)
    sweep_plot = sns.relplot(data=sweep_df, x='lr', y='val_loss')
    sweep_plot.set(title=f"{model_type} hyperparameter sweep")
    sweep_plot.set(ylim=(1e23, 1e25))
    sweep_plot.fig.savefig(f'plots/{model_type}_sweep.png', dpi=400)
    plt.close(plt.gcf())

def compare_baseline_to_model(baseline_csv, model_csv):
    baseline_loss = pd.read_csv(baseline_csv, index_col=0)
    model_loss = pd.read_csv(model_csv, index_col=0)
    new_df = baseline_loss[['epoch', 'val']]
    new_df['model_val'] = model_loss['val']
    loss_df = pd.melt(new_df, id_vars=['epoch'], value_vars=['val','model_val'], var_name='model type', value_name='validation MSE')
    new_names = {'val': 'baseline', 'model_val': 'model'}
    loss_df['model type'] = loss_df['model type'].map(new_names)
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='validation MSE', hue='model type', kind='line').set(title=f"Edge features improve performance")
    plt.tight_layout()
    loss_plot.fig.savefig(f'plots/comparison_loss.png', dpi=400)


#compare_baseline_to_model(f'results/baseline_0.05_{Hyperparams.n_epochs}_train.csv', f'results/model_0.05_{Hyperparams.n_epochs}_train.csv')
loss_plot(f"baseline_{hyperparams['learning_rate']}_1000")
for model_type in ['baseline', 'model']:
    for e in range(0, hyperparams['n_epochs'], hyperparams['save_model_interval']):
        pred_plot(f"results/{model_type}_{hyperparams['learning_rate']}_{e}_out_of_{hyperparams['n_epochs']}_prediction.csv",\
        f"{model_type} prediction after {e} epochs",\
        f"plots/{model_type}_{hyperparams['n_epochs']}_{e}.png")
    pred_plot(f"results/{model_type}_{hyperparams['learning_rate']}_prediction.csv",\
        f"{model_type} prediction after {hyperparams['n_epochs']} epochs", f"plots/{model_type}_{hyperparams['learning_rate']}.png")




