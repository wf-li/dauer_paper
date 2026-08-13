import numpy as np
import json
from pathlib import Path
from matplotlib.colors import ListedColormap
from plotter import Plotter

pca_root = Path('../analysis_modules/pca')
pca_components = 2

dataset_selection = {
    'connectome_all': [
        'L1-1','L1-2','L1-3','L1-4',
        'L2','L3','adult-1','adult-2',
        'dauer-1','dauer-2','dauer-daf2'
    ],
    'connectome': [
        'L2','L3','adult-1','adult-2',
        'dauer-1','dauer-2','dauer-daf2'
    ],
    'proximity': [
        'L2','L3','adult-2',
        'dauer-1','dauer-2','dauer-3'
    ],
    'drive': [
        'L2','L3','adult-2','dauer-1','dauer-2'
    ]
}

output_path = Path('figures/pca')
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {output_path}")

for datatype in dataset_selection.keys():
    input_file = Path(f'pca_{datatype}.json')
    input_path = pca_root / input_file
    use_kmeans_labels=True

    with open(input_path, 'r') as f:
        data = json.load(f)
    data['features'] = np.array(data['features'])
    labels = None
    # labels = data['labels']

    cmap = ListedColormap(['k','#FF0000'])
    s= 100

    xlabel = f'PC1 ({data['explained_variance'][0]:.2%} variance)'
    ylabel = f'PC2 ({data['explained_variance'][1]:.2%} variance)'


    figure_generator = Plotter(output_path='figures/pca')
    figure_generator.plot_pca(
        data['features'][:,0],
        data['features'][:,1],
        xlabel,
        ylabel,
        labels=labels,
        clabels=data['cluster_labels'] if use_kmeans_labels else None,
        cmap=cmap,
        save_as=f'pca_{datatype}',
    )