import json
import numpy as np
from .pca import run_pca
from ..src.data_manager import DataManager
from ..src.utils import exclude_postemb_edgetable
from pathlib import Path
from pathlib import Path
import json
import numpy as np
from analysis_modules.pca.pca import run_pca
from analysis_modules.src.utils import exclude_postemb_edgetable
from analysis_modules.src.data_manager import DataManager

# datatype = 'connectome'
pca_root = Path('analysis_modules/pca/outputs')
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

for datatype in dataset_selection.keys():
    ds = dataset_selection[datatype]

    dm = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=True,
        npair_result=True
    )

    if datatype == 'drive':
        edgetable = dm.get_drive_edgetable(consistent=True)
    elif datatype == 'connectome_all':
        edgetable = dm.get_data_edgetable('connectome')
    else:
        edgetable = dm.get_data_edgetable(datatype)
    edgetable = edgetable.loc[:,ds]

    if datatype == 'connectome':
        edgetable = np.sqrt(edgetable)
    if datatype == 'proximity':
        edgetable[edgetable<10]=0
        edgetable = edgetable.replace(0,np.nan)
        edgetable = np.log(edgetable)

    # exclude postembryonic neurons from index
    edgetable = exclude_postemb_edgetable(edgetable, ds)

    edgetable = edgetable.fillna(0)
    edgetable = edgetable[edgetable.sum(axis=1) > 0]

    output_data = run_pca(edgetable, pca_components)

    output_file = Path(f'pca_{datatype}.json')
    with open(pca_root / output_file, 'w+') as fp:
        json.dump(output_data, fp, indent=4)
