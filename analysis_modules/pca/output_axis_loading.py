import json
from pathlib import Path
from .pca_axis_loading import get_adjusted_loadings

classification_path = Path('data/connectomes/metadata/synapse_neighborhood_classification_table.csv')

for datatype in ['connectome','proximity','drive','connectome_2_dauer','proximity_2_dauer']:
    pca_path = Path('analysis_modules/pca/outputs')
    pca_file = Path(f'pca_{datatype}.json')
    with open(pca_path / pca_file, 'r') as f:
        pca_data = json.load(f)
    loading_df = get_adjusted_loadings(pca_data)
    loading_df.to_csv(pca_path / f'{datatype}_loadings.csv')