import numpy as np
import pandas as pd
from plotter import Plotter
import json

feed_colors = {'Feed-forward': '#C7EAE4', 'Feed-back': '#EAC9C1', 'Recurrent': 'white'}

data = pd.read_csv('analysis_modules/feedforward/outputs/edgelist_ffclass.csv', index_col=[0,1])

with open('plotting/utils/metadata.json', 'r') as f:
    metadata = json.load(f)
    
# sum each edge class
edge_class_sums = data.groupby('edge_class').sum()
edge_class_proportions = edge_class_sums / edge_class_sums.sum()

plot_data = edge_class_proportions.T
colors = [feed_colors[edge_class] for edge_class in plot_data.columns]

y_nondauer = []
x_nondauer = []
y_dauer = []
x_dauer = []

for dataset_name, meta in metadata.items():
    if dataset_name not in plot_data.index:
        print(f"{dataset_name} not found, continuing...")
        continue
    if meta['stage'] == 'dauer':
        y_dauer.append(plot_data.loc[dataset_name].values)
        x_dauer.append(meta['age_visual'])
    else:
        y_nondauer.append(plot_data.loc[dataset_name].values)
        x_nondauer.append(meta['age_visual'])

y_nondauer = np.array(y_nondauer).T
y_dauer = np.array(y_dauer).T

ymin = 0
ymax = 0.6
ylabel = f'Proportion of synapses'
yticks = np.arange(0,ymax+0.1, 0.2)

# --- Plotting ---
figure_generator = Plotter(output_path='figures')
figure_generator.plot_multiple_xy(
    xs=x_nondauer,
    ys=y_nondauer,
    x_dauers=x_dauer,
    y_dauers=y_dauer,
    ymin=ymin,
    ymax=ymax,
    ylabel=ylabel,
    yticks=yticks,
    save_as = f'feedforward',
    best_fit_line=True,
    show_plot=True,
    colors=colors
)