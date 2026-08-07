import pandas as pd
import numpy as np

def sum_dfs(dfs, datatype):
    summed_df = None
    for df in dfs:
        if datatype == 'contactome':
            df = np.log10(df[df > 0])
            df[np.isnan(df)] = 0
        if summed_df is None:
            summed_df = df
        else:
            summed_df = summed_df.add(df, fill_value=0)
    return summed_df/len(dfs)

def get_adj_shared(dfs):
    ''' Applies set logic; returns bool matrix of same shape as input matrix
    '''
    df_shared = dfs[0].astype(bool)
    for df in dfs[1:]:
        df_shared &= df.astype(bool)
    return df_shared

datasets = [
    # 'dataset1','dataset2','dataset3','dataset4',
    'dataset5','dataset6','dataset7','dataset8',
    # 'dauer-1','dauer-2','dauer-daf2'
]
datatype = 'connectome'

dfs = []
neuron_order = pd.read_csv('data/raw/contactome_dataset8.csv', index_col = 0).index

for dataset in datasets:
    csv_file = f'data/raw/{datatype}_{dataset}.csv' 
    dfs.append(pd.read_csv(csv_file, index_col = 0))

df_shared = get_adj_shared(dfs)

result = sum_dfs(dfs, datatype)#.loc[neuron_order,neuron_order]
result = result[df_shared]
print(result.head())
result.to_csv(f'data/processed/mean_{datatype}_L2L3Adult_intersection.csv')