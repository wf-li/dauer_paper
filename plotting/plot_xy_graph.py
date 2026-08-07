import pandas as pd
import numpy as np
import json
from plotter import Plotter

file_path = 'analysis_modules/dataset_stats/outputs/dataset_summary_statistics.csv'
parameter_path = 'plotting/utils/xy_graph_parameters.csv'
metadata_path = 'plotting/utils/metadata.json'

# Load and process the data
data = pd.read_csv(file_path, index_col=0)
params = pd.read_csv(parameter_path)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

plot_names = {
    'L1-1': 'L1-0hr',
    'L1-2': 'L1-5hr',
    'L1-3': 'L1-8hr',
    'L1-4': 'L1-16hr',
    'L2': 'L2',
    'L3': 'L3',
    'adult-1': 'Adult 1',
    'adult-2': 'Adult 2',
    'dauer-1': 1,
    'dauer-2': 2,
    'dauer-3': 3,
    'dauer-daf2': 4
}

data['plot_name'] = data.index.map(plot_names)
data['type'] = data.index.map(lambda x: 'dauer' if metadata[x]['stage']=='dauer' else 'nondauer')
data['timepoint'] = data.index.map(lambda x: metadata[x]['age_hours'] if metadata[x]['stage']!='dauer' else np.nan)

# plot_metric = 'synapse_over_contact_surface'
for plot_metric in params['variable_name'].values:
    filt = params['variable_name']==plot_metric
    ymin = params[filt]['y_axis_min'].values[0]
    ymax = params[filt]['y_axis_max'].values[0]
    ylabel = params[filt]['ylabel'].values[0]
    yticks = np.linspace(ymin, ymax, params[filt]['n_yticks'].values[0])

    if plot_metric == 'cable_length':
        data[plot_metric] = (data[plot_metric])/1000000
    if plot_metric == 'contact_surface':
        data[plot_metric] = (data[plot_metric])/1000000
        ylabel = f'{ylabel} (um$^2$)'
    if plot_metric == 'volume':
        data[plot_metric] = (data[plot_metric])/1000000000
        ylabel = f'{ylabel} (um$^3$)'
    # if plot_metric=='synapse_over_contact':
    #     data['synapse_over_contact'] = data['synapses']/data['contacts']
    if plot_metric=='synapses_over_cable_length':
        # data['synapse_over_contact_surface'] = data['synapses']/(data['contact_surface']/1000000)
        ylabel = 'Synapse/Neurite Length (um)'
    if plot_metric=='synapses_over_contact_area':
        # data['synapse_over_contact_surface'] = data['synapses']/(data['contact_surface']/1000000)
        ylabel = 'Synapse/Contact Area (um$^2$)'

    # Filter for control data
    filt_nondauer = (data['type'] == 'nondauer') & ~np.isnan(data[plot_metric])
    x_nondauer = data[filt_nondauer]['timepoint'].values
    y_nondauer = data[filt_nondauer][plot_metric].values

    # Filter for dauer data
    filt_dauer = (data['type'] == 'dauer') & ~np.isnan(data[plot_metric])
    x_dauer = data[filt_dauer]['plot_name'].values.astype(int)
    y_dauer = data[filt_dauer][plot_metric].values

    # --- Plotting ---
    figure_generator = Plotter(output_path='figures/dataset_stats')
    figure_generator.plot_xy_graph(
        x=x_nondauer,
        y=y_nondauer,
        x_dauer=x_dauer,
        y_dauer=y_dauer,
        ymin=ymin,
        ymax=ymax,
        ylabel=ylabel,
        yticks=yticks,
        save_as = f'{plot_metric}_plot',
        connect_points=True,
        dauer_lines=True
    )