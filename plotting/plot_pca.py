import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plotter import Plotter

plot_metric = 'connectome'
output_path = f'figures/pca'

input_path = f'analysis_modules/pca_contribution/outputs/pca_{plot_metric}.json'
metadata_path = 'plotting/utils/metadata.json'
use_kmeans_labels=False

with open(input_path, 'r') as f:
    data = json.load(f)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

data['features'] = np.array(data['features'])

if use_kmeans_labels:
    clabels = data['cluster_labels']
else:
    clabels = [
        1 if metadata[label]['stage'] == 'dauer' else 0 for label in data['labels'] 
    ]
labels = [
    label if metadata[label]['stage'] == 'L1' else metadata[label]['plot_name'] for label in data['labels'] 
]

cmap = ListedColormap(['k','#90D5FF'])
s= 100

xlabel = f'PC1 ({data['explained_variance'][0]:.2%} variance)'
ylabel = f'PC2 ({data['explained_variance'][1]:.2%} variance)'

plotter = Plotter(output_path = output_path)
plotter.plot_pca(
    data['features'][:, 0], data['features'][:, 1],
    xlabel, ylabel, labels, save_as = f'pca_{plot_metric}',
    cmap = cmap, clabels = clabels, size = s, show_plot=True
)