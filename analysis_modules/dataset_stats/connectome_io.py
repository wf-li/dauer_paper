import pandas as pd
from pathlib import Path
from ..src.neuron_info import ntype
from ..src.data_manager import DataManager

OUTPUT_DIR = 'analysis_modules/dataset_stats/outputs/'

def get_connectome_io(adj_mats, method='synapses'):
    """
    Loads connectome input/output data.
    """
    assert method in ['synapses', 'connections']

    dfs = {}

    for name, df in adj_mats.items():
        if method == 'synapses':
            outputs = df.sum(axis=0).rename(f'{name}_output')
            inputs = df.sum(axis=1).rename(f'{name}_input')
        elif method == 'connections':
            outputs = df.astype(bool).sum(axis=0).rename(f'{name}_output')
            inputs = df.astype(bool).sum(axis=1).rename(f'{name}_input')
        else:
            raise ValueError("Invalid method. Choose 'synapses' or 'connections'.")
        total = (inputs + outputs).rename(f'{name}_total')

        # Store results for this file
        dfs[name] = pd.concat([outputs, inputs, total], axis=1)

    combined_df = pd.concat(dfs.values(), axis=1)

    result = {
        'output': combined_df[[c for c in combined_df.columns if 'output' in c]].copy(),
        'input': combined_df[[c for c in combined_df.columns if 'input' in c]].copy(),
        'total': combined_df[[c for c in combined_df.columns if 'total' in c]].copy(),
    }
    
    for key in result:
        result[key].columns = [col.replace(f"_{key}", "") for col in result[key].columns]

    return result

if __name__ == "__main__":
    method = 'synapses' # connections or synapses

    data = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=False,
        npair_result=True,
    )

    adj_mats = data.get_data_adj()

    connectome_io = get_connectome_io(
        adj_mats,
        method
    )

    for k, v in connectome_io.items():
        v['type'] = v.index.map(ntype)
        fname = Path(OUTPUT_DIR) / (f"{k}_{method}.csv")
        v.to_csv(fname)