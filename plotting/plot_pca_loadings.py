import json
import numpy as np
from plotter import Plotter

datatype = 'contactome'
plot_type = 'both' # circular, linear  or both
plot_component = 'PC1' # PC1, PC2, or both
n_features_linear = 40
n_features_circular = 20

output_path = 'figures/pca'
output_file_linear = f"loadings_{datatype}_{plot_component}"
output_file_circle = f'{datatype}_loadings_circular_{plot_component}'

input_path = f'analysis_modules/pca_contribution/outputs/pca_{datatype}.json'

with open(input_path, 'r') as f:
    data = json.load(f)

loadings = np.array(data['loadings'])
feature_labels = np.array(data['edge_labels'])

if plot_component == 'PC1':
    plot_pc = 0
    separate_circular_pc = True
elif plot_component == 'PC2':
    plot_pc = 1
    separate_circular_pc = True
elif plot_component == 'both':
    if plot_type == 'linear':
        raise ValueError("plot_type 'linear' plots only PC1 or PC2, not both")
    plot_pc=[0,1]
    separate_circular_pc = False

plotter = Plotter(
    output_path=output_path
)
if plot_type in ['circular', 'both']:
    plotter.plot_pca_loadings_circular(
        loadings, feature_labels, n=n_features_circular,
        separate=separate_circular_pc,
        pcs_to_consider=plot_pc,
        cmap = 'Spectral',
        save_as=output_file_circle,
        show_plot=True
    )
if plot_type in ['linear', 'both']:
    plotter.plot_pca_loadings_linear(
        loadings, feature_labels,
        n_features_linear, plot_pc,
        save_as=output_file_linear, show_plot=True
    )