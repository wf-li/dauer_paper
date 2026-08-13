import pandas as pd
from pathlib import Path
from plotter import Plotter

datatype = 'connectome'

loading_path = Path('analysis_modules/pca/outputs')
loading_file = Path(f'{datatype}_loadings.csv')
classification_path = Path('data/connectomes/metadata/synapse_neighborhood_classification_table.csv')

loading_df = pd.read_csv(loading_path / loading_file, index_col=[0,1])

acceptable_classes = [
    'dauer_increased','dauer_decreased','maintained',#'variable'
]
class_data = pd.read_csv(classification_path, index_col=[0,1])

acceptable_index = class_data[class_data['connectivity_3_dauer'].isin(acceptable_classes)].index
overlap = loading_df.index.intersection(acceptable_index)

results_subset = loading_df.loc[overlap]

loadings = results_subset['separation_axis_loading'].to_numpy()
labels   = class_data.loc[overlap, 'connectivity_3_dauer'].to_numpy()

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
bin_width  = data_range * 0.005

figure_generator = Plotter(output_path = 'figures/pca_loading')
figure_generator.plot_pca_loadings(
    loadings,
    bin_width,
    classes = labels,
    class_order = acceptable_classes,
    save_as=f'{datatype}_loadings',
    show_plot=True,
    flip=True
)
