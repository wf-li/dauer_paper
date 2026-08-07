import pandas as pd
from plotter import Plotter

dark_colors = {
    'sensory': '#fda0fd',
    'inter': '#ff442f',
    'motor': '#5cafff',
    'modulatory': '#ffc000',
    # 'muscle': '#7ec95a',
}

filter_zeroes = True

io_type = 'output' # input, output
plot_metric = 'connections' # synapses, connections

ylabel = f'Proportion of {io_type} {plot_metric}'

df = pd.read_csv(f'analysis_modules/dataset_stats/outputs/{io_type}_{plot_metric}.csv', index_col=0)

type_data = df.groupby('type').sum()
plot_data = type_data/(type_data.sum(axis=0))
order = ['sensory', 'inter', 'motor', 'modulatory']
plot_data = plot_data.sort_index(key=lambda x: x.map({val: i for i, val in enumerate(order)}))

# --- Plotting ---
figure_generator = Plotter(output_path='figures/dataset_stats')
figure_generator.plot_stacked_area(
    plot_data,
    ylabel=ylabel,
    colors = [v for v in dark_colors.values()],
    save_as = f'{io_type}_{plot_metric}_stacked_area',
    show_plot=True
)