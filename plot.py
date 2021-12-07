import seaborn as sns
import pandas as pd

sns.set_theme()

def loss_plot(model_type):
    loss_df = pd.read_csv(f'results/{model_type}_train.csv', index_col=0)
    loss_df = loss_df.melt('epoch', var_name='loss_type', value_name='loss')
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', hue='loss_type', kind='line').set(title=f"{model_type} training trajectory")
    loss_plot.fig.savefig(f'plots/{model_type}_loss.png', dpi=400)

def pred_plot(model_type, bounding_scaling_factor=.15):
    preds_df = pd.read_csv(f'results/{model_type}_prediction.csv')
    # make limits nicely surround points
    min_val = preds_df[['ground_truth','prediction']].min().values.min() * (1 - bounding_scaling_factor)
    max_val = preds_df[['ground_truth', 'prediction']].max().values.max() * (1 + bounding_scaling_factor)
    lim = (min_val, max_val)
    preds_plot = sns.relplot(data=preds_df, x='ground_truth', y='prediction')
    preds_plot.set(xscale='log', yscale='log') # large spread in values
    preds_plot.set(xlabel='actual GDP', ylabel='predicted GDP') # label axes
    preds_plot.set(xlim=lim, ylim=lim) # limits must be same to match intution that y=x is correct
    preds_plot.set(title=f"{model_type} prediction error")
    preds_plot.fig.savefig(f'plots/{model_type}_prediction_error.png', dpi=400)

for model_type in ['baseline', 'model']:
    loss_plot(model_type)
    pred_plot(model_type)
