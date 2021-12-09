from numpy.core.fromnumeric import size
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util import *
from hyperparams import hyperparams

sns.set_theme()

def loss_plot(model_type):
    '''
    Plot the trajectory of a model's training from a dataframe that has train/val loss at each epoch. 
    '''
    plt.yscale('log')
    loss_df = pd.read_csv(loss_file(model_type), index_col=0)
    plt.plot(loss_df['epoch'], loss_df['train'], '-r', label='train')
    plt.plot(loss_df['epoch'], loss_df['val'], '-b', label='val')
    plt.legend(loc='upper right', title='loss type')
    plt.yscale('log')
    plt.ylabel('log MSE')
    plt.xlabel('epoch')
    plt.title(f'{model_type} training trajectory')
    plt.savefig(f'plots/{model_type}_loss.png', dpi=400)
    plt.close(plt.gcf())

def pred_plot(preds_df, title, out_file, max_x, max_y):
    '''
    Given a DataFrame that has ground truth and predicted values, plot a scatter
    of true vs. predicted values. Force scaling via max_x and max_y s.t. we have
    a series of plots with the same axes. Also plot line showing a hypothetical perfect
    model's prediction.
    '''
    print(f"Making rediction plot at {out_file}...", end='')
    plt.yscale('linear') # re-set to linear axis (data is already log-ed)
    plt.plot([0, max_x], [0, max_x], color='red', zorder=1) # place behind points
    plt.scatter(preds_df['ground_truth'], preds_df['prediction'], s=.7, zorder=2) # place in front of line
    plt.subplots_adjust(left=0.15, top=0.9) # prevent x-label and title from getting cut off
    plt.xlabel('actual log_GDP')
    plt.ylabel('predicted log_GDP') # label axes
    plt.xlim((0, max_x))
    plt.ylim((0, max_y))
    plt.title(title)
    plt.savefig(out_file, dpi=400)
    plt.close(plt.gcf())
    print("done!")

def compare_baseline_to_model():
    baseline_loss = pd.read_csv(loss_file('baseline'), index_col=0)
    model_loss = pd.read_csv(loss_file('model'), index_col=0)
    plt.yscale('log')
    plt.plot(baseline_loss['epoch'], baseline_loss['val'], '-r', label='baseline')
    plt.plot(model_loss['epoch'], model_loss['val'], '-b', label='model')
    plt.legend(loc='upper right', title='model type')
    plt.ylabel('log validation MSE')
    plt.title('Edge features improve performance')
    plt.tight_layout()
    plt.savefig(f'plots/comparison_loss.png', dpi=400)

# Now, make the plots!
compare_baseline_to_model()
for model_type in ['baseline', 'model']:
    epoch_range = range(0, hyperparams['n_epochs'] + 1, hyperparams['save_model_interval'])

    # read in all the prediction files
    dfs = [(e, pd.read_csv(preds_file(model_type, e), index_col=0)) for e in epoch_range]

    # get limits such that every model's predictions < max_y and every model's ground_truth < max_x
    max_y = max([df[['prediction']].max().values.max() for _, df in dfs])
    max_x = max([df[['ground_truth']].max().values.max() for _, df in dfs])

    # give some breathing room so that axes don't cut off points
    max_y *= 1.05
    max_x *= 1.05
    for e, df in dfs:
        pred_plot(df, f"{model_type} on test data after {e} epochs", preds_plot_file(model_type, e), max_x, max_y)
    loss_plot(model_type)