import pandas as pd
from ..src.data_manager import DataManager

OUTPUT_DIR = 'analysis_modules/dataset_stats/outputs/'

include_postemb = True
include_muscle = True
npair_result = True

data = DataManager(
    data_path='data',
    include_postemb=include_postemb,
    include_muscle=include_muscle,
    npair_result=npair_result,
)

# connectome and contactome with muscles
G = data.get_data_edgelist(type='connectome')
G_contacts = data.get_data_edgelist(type='contactome')

# cable length calculated without muscles
data.include_muscle = False

stats = {
    'connections': G.astype(bool).sum(axis=0),
    'synapses': G.sum(axis=0),
    'contacts': G_contacts.astype(bool).sum(axis=0),
    'contact_surface': G_contacts.sum(axis=0),
    'cable_length': data.get_cable_lengths(),
    'volume': data.get_volume()
}

# other stats
stats['synapses_over_connections'] = stats['synapses']/stats['connections']
stats['synapses_over_contact_area'] = stats['synapses']/(stats['contact_surface']/1000000)
stats['synapses_over_cable_length'] = stats['synapses']/(stats['cable_length']/1000)

pd.concat(stats, axis=1).to_csv(OUTPUT_DIR + 'dataset_summary_statistics.csv')