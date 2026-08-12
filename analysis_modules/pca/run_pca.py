import json
import numpy as np
from .pca import run_pca
from ..src.data_manager import DataManager
from ..src.utils import exclude_postemb_edgetable
from pathlib import Path

datatype = 'connectome'
output_path = Path('analysis_modules/pca/outputs/')
output_file = Path(f'pca_{datatype}.json')
pca_components = 2
random_seed = 47
kmeans_clusters = 2

data = DataManager(
    data_path='data',
    include_postemb=True,
    include_muscle=True,
    npair_result=True
)

edgetable = data.get_data_edgetable(datatype=datatype)

# include datasets
datasets = [
    'L1-1','L1-2','L1-3','L1-4',
    'L2','L3','adult-1','adult-2',
    'dauer-1','dauer-2','dauer-daf2'
]
edgetable = edgetable.loc[:,datasets]
edgetable = edgetable.replace(np.nan,0)

# exclude postembryonic neurons from index
edgetable = exclude_postemb_edgetable(edgetable, datasets)
features = np.array([edgetable[col].values for col in edgetable.columns])
output_data = run_pca(edgetable)

output_path.mkdir(parents = True, exist_ok=True)
with open(output_path / output_file, 'w+') as fp:
    json.dump(output_data, fp, indent=4)
