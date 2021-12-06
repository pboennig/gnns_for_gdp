import seaborn as sns
import pandas as pd

sns.set_theme()

def loss_plot(model_type):
    loss_df = pd.read_csv(f'results/{model_type}_train.csv', index_col=0)
    loss_df = loss_df.melt('epoch', var_name='loss_type', value_name='loss')
    loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', hue='loss_type', kind='line').set(title=f"{model_type} training trajectory")
    loss_plot.fig.savefig(f'plots/{model_type}_loss.png', dpi=400)

def pred_plot(input_csv, output_file):
    preds_df = pd.read_csv('results/prediction.csv')
    lim = (preds_df[['ground_truth','prediction']].min().values.min() - 1e6, preds_df[['ground_truth', 'prediction']].max().values.max())
    preds_plot = sns.relplot(data=preds_df, x='ground_truth', y='prediction').set(xscale='log', yscale='log', xlabel='actual GDP', ylabel='predicted GDP', xlim=lim, ylim=lim)
    preds_plot.fig.savefig('plots/prediction_error.png', dpi=400)

loss_plot('baseline')
