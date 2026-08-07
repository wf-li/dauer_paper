import pandas as pd
from ..src.data_manager import DataManager
from ..src.neuron_info import ntype

def feed_type(edge, only_sensory=False, with_muscle=False):
    edge_type = (ntype(edge[0]), ntype(edge[1]))
    if only_sensory:
        if edge_type in (('sensory', 'inter'), ('sensory', 'motor'), ('sensory', 'modulatory')):
            return 'Feed-forward'
        if edge_type in (('inter', 'sensory'), ('motor', 'sensory'), ('modulatory', 'sensory')):
            return 'Feed-back'
        if edge_type in (('sensory', 'sensory'), ):
            return 'Recurrent'
        return None
    if edge_type in (('sensory', 'inter'), ('inter', 'motor'), ('sensory', 'motor'), ('modulatory', 'inter'), ('sensory', 'modulatory'), ('modulatory', 'motor')):
        return 'Feed-forward'
    if edge_type in (('inter', 'sensory'), ('motor', 'inter'), ('motor', 'sensory'), ('inter', 'modulatory'), ('modulatory', 'sensory')):
        return 'Feed-back'
    if edge_type in (('sensory', 'sensory'), ('inter', 'inter'), ('motor', 'motor'), ('modulatory', 'modulatory')):
        return 'Recurrent'
    if with_muscle and edge_type[1] == 'muscle':
        return 'Feed-forward'
    return None

if __name__ == "__main__":
    fname = 'edgelist_ffclass.csv'
    OUTPUT_PATH = f'analysis_modules/feedforward/outputs/{fname}'

    data = DataManager(
        data_path='data',
        include_postemb=False,
        include_muscle=False,
        npair_result=True
    )

    edge_data = data.get_data_edgelist()

    edge_data['edge_class'] = edge_data.index.to_series().apply(
        lambda x: feed_type(x, only_sensory=False, with_muscle=False)
    )

    edge_data.to_csv(OUTPUT_PATH)