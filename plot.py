from numpy.core.fromnumeric import size
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util import *
from hyperparams import hyperparams

sns.set_theme()

def loss_plot(model_type):
    plt.subplots_adjust(top=.8)
    loss_df = pd.read_csv(loss_file(model_type), index_col=0)
    loss_df = loss_df.melt('epoch', var_name='loss_type', value_name='loss')
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', hue='loss_type', kind='line').set(title=f"{model_type} training trajectory")
    plt.yscale('log')
    loss_plot.fig.savefig(f'plots/{model_type}_loss.png', dpi=400)
    plt.close(plt.gcf())

def pred_plot(preds_df, title, out_file, max_x, max_y):
    print(f"Making rediction plot at {out_file}...", end='')
    # make limits nicely surround points
    plt.plot([0, max_x], [0, max_x], color='red', zorder=1)
    plt.scatter(preds_df['ground_truth'], preds_df['prediction'], s=.7, zorder=2)
    plt.subplots_adjust(left=0.15, top=0.9) # prevent x-label and title from getting cut off
    #preds_plot.set(xscale='log', yscale='log') # large spread in values
    plt.xlabel('actual log_GDP')
    plt.ylabel('predicted log_GDP') # label axes
    plt.xlim((0, max_x))
    plt.ylim((0, max_y))
    plt.title(title)
    #plt.tight_layout()
    plt.savefig(out_file, dpi=400)
    plt.close(plt.gcf())
    print("done!")

def hyperparams_plot(model_type):
    sweep_df = pd.read_csv(f'results/{model_type}_hyperparams.csv', index_col=0)
    sweep_plot = sns.relplot(data=sweep_df, x='lr', y='val_loss')
    sweep_plot.set(title=f"{model_type} hyperparameter sweep")
    sweep_plot.set(ylim=(1e23, 1e25))
    sweep_plot.fig.savefig(f'plots/{model_type}_sweep.png', dpi=400)
    plt.close(plt.gcf())

def compare_baseline_to_model():
    baseline_loss = pd.read_csv(loss_file('baseline'), index_col=0)
    model_loss = pd.read_csv(loss_file('model'), index_col=0)
    new_df = baseline_loss[['epoch', 'val']]
    new_df['model_val'] = model_loss['val']
    loss_df = pd.melt(new_df, id_vars=['epoch'], value_vars=['val','model_val'], var_name='model type', value_name='validation MSE')
    new_names = {'val': 'baseline', 'model_val': 'model'}
    loss_df['model type'] = loss_df['model type'].map(new_names)
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='validation MSE', hue='model type', kind='line').set(title=f"Edge features improve performance")
    loss_plot.set(yscale='log')
    plt.tight_layout()
    loss_plot.fig.savefig(f'plots/comparison_loss.png', dpi=400)


compare_baseline_to_model()

for model_type in ['baseline', 'model']:
    #epoch_range = range(0, hyperparams['n_epochs'] + 1, hyperparams['save_model_interval'])
    epoch_range = range(0, hyperparams['n_epochs'] + 1, hyperparams['save_model_interval'])
    dfs = [(e, pd.read_csv(preds_file(model_type, e), index_col=0)) for e in epoch_range]
    max_y = max([df[['prediction']].max().values.max() for _, df in dfs])
    max_x = max([df[['ground_truth']].max().values.max() for _, df in dfs])
    max_y *= 1.1 # don't cut off values
    max_x *= 1.1 # don't cut off values
    for e, df in dfs:
        pred_plot(df, f"{model_type} on test data after {e} epochs", preds_plot_file(model_type, e), max_x, max_y)

    loss_plot(model_type)

    




