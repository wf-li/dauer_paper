import pandas as pd
from pathlib import Path
from plotter import Plotter

# datatype = 'connectome'

for datatype in ['connectome','proximity']:
    loading_path = Path('analysis_modules/pca/outputs')
    loading_file = Path(f'{datatype}_loadings.csv')
    loading_df = pd.read_csv(loading_path / loading_file, index_col=[0,1])

    classification_path = Path('data/connectomes/metadata/synapse_neighborhood_classification_table.csv')
    class_data = pd.read_csv(classification_path, index_col=[0,1])

    plot_classes = {
        'connectome': {
            'classes': ['dauer_increased','dauer_decreased','maintained'],
            'df_column': 'connectivity_3_dauer'
        },
        'proximity':{
            'classes': ['dauer_increased','dauer_decreased','maintained'],
            'df_column': 'neighborhood_3_dauer'
        },
        'connectome_all':{
            'classes': ['dauer_increased','dauer_decreased','maintained','variable'],
            'df_column': 'connectivity_3_dauer'
        }
    }

    plot_index = class_data[class_data[plot_classes[datatype]['df_column']].isin(plot_classes[datatype]['classes'])].index
    overlap = loading_df.index.intersection(plot_index)

    results_subset = loading_df.loc[overlap]

    loadings = results_subset['separation_axis_loading'].to_numpy()
    labels   = class_data.loc[overlap, plot_classes[datatype]['df_column']].to_numpy()

    color_map = {
        'dauer_increased': '#ed2024',
        'maintained': 'black',
        'dauer_decreased': '#abdbee',
        'variable': 'grey',
        'late_postembryonic': 'grey',
        'no_synapse': 'grey',
        'nan': 'grey'
    }

    data_range = loadings.max() - loadings.min()
    bin_width  = data_range * 0.01

    figure_generator = Plotter(output_path = 'figures/pca_loading')
    figure_generator.plot_pca_loadings(
        loadings,
        bin_width,
        classes = labels,
        class_order = plot_classes[datatype]['classes'],
        color_map = color_map,
        save_as=f'{datatype}_loadings',
        show_plot=True,
        flip=False,
        figsize=(5,3),
        alpha=1
    )
