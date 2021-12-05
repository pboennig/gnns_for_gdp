import seaborn as sns
import pandas as pd

sns.set_theme()

loss_df = pd.read_csv('results/baseline_train.csv')
loss_plot = sns.relplot(data=loss_df, x='epoch', y='loss', kind='line').set(title="Baseline training trajectory")
loss_plot.fig.savefig('plots/baseline_loss.png', dpi=400)

preds_df = pd.read_csv('results/prediction.csv')
lim = (preds_df[['ground_truth','prediction']].min().values.min() - 1e6, preds_df[['ground_truth', 'prediction']].max().values.max())
preds_plot = sns.relplot(data=preds_df, x='ground_truth', y='prediction').set(xscale='log', yscale='log', xlabel='actual GDP', ylabel='predicted GDP', xlim=lim, ylim=lim)
preds_plot.fig.savefig('plots/prediction_error.png', dpi=400)
