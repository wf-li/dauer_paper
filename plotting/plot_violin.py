import pandas as pd
from plotter import Plotter
import json

filter_zeroes = True

plot_metric = 'connections' # synapses, connections

ymin = 0
ymax = 1200
ylabel = f'{plot_metric} per neuron'
yticks = list(range(ymin, ymax+1, 300))

df = pd.read_csv(f'analysis_modules/dataset_stats/outputs/total_{plot_metric}.csv', index_col=0)
with open('plotting/utils/metadata.json', 'r') as f:
    metadata = json.load(f)

data_nondauer = []
x_nondauer = []
data_dauer = []
x_dauer = []

for dataset_name, meta in metadata.items():
    if dataset_name == 'type':
        continue
    if dataset_name not in df.columns:
        print(f"{dataset_name} not found, continuing...")
        continue
    if meta['stage'] == 'dauer':
        if filter_zeroes:
            data_dauer.append(df[dataset_name][df[dataset_name]>0].values)
        else:
            data_dauer.append(df[dataset_name].values)
        x_dauer.append(meta['age_visual'])
    else:
        if filter_zeroes:
            data_nondauer.append(df[dataset_name][df[dataset_name]>0].values)
        else:
            data_nondauer.append(df[dataset_name].values)
        x_nondauer.append(meta['age_visual'])

# --- Plotting ---
figure_generator = Plotter(output_path='figures/dataset_stats')
figure_generator.plot_violin_graph(
    data_nondauer = data_nondauer,
    x_nondauer = x_nondauer,
    data_dauer = data_dauer,
    x_dauer=x_dauer,
    ymin=ymin,
    ymax=ymax,
    ylabel=ylabel,
    yticks=yticks,
    save_as = f'{plot_metric}_plot'
)