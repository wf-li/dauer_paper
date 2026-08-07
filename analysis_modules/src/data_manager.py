import re
import json
import pandas as pd
from pathlib import Path
from ..src.neuron_info import ntype, npair, is_postemb

rename_dataset_dict = {
    'dataset1': 'L1-1',
    'dataset2': 'L1-2',
    'dataset3': 'L1-3',
    'dataset4': 'L1-4',
    'dataset5': 'L2',
    'dataset6': 'L3',
    'dataset7': 'adult-1',
    'dataset8': 'adult-2',
    'dauer-1': 'dauer-1',
    'dauer-2': 'dauer-2',
    'dauer-FIB': 'dauer-3',
    'dauer-daf2': 'dauer-daf2',
}

class DataManager:
    def __init__(self, data_path, **kwargs):
        """
        Initializes the DataManager.

        Args:
            data_path (str or Path): Path to the root data directory.
            **kwargs:
                - include_postemb (bool): Whether to include postembryonic neurons. Default: True
                - include_muscle (bool): Whether to include muscle cells. Default: False
                - npair_result (bool): Whether to pair neurons. Default: False
                - replacements (dict): Dictionary of neuron names to replace. Default: {}
                - rename_ds (bool): Whether to replace dataset names. Default: True
        """
        self.data_path = Path(data_path)
        self.include_postemb = kwargs.get('include_postemb', True)
        self.include_muscle = kwargs.get('include_muscle', False)
        self.npair_result = kwargs.get('npair_result', False)
        self.replacements = kwargs.get('replacements', {})
        self.rename_ds = kwargs.get('rename_ds', True)

        self.dirs = {
            'connectome': self.data_path / 'connectomes' / 'individual_cells',
            'contactome': self.data_path / 'contactomes' / 'individual_cells',
            'proximity': self.data_path / 'proximity',
            'proximity_points_counts': self.data_path / 'proximity_points' / 'counts',
            'proximity_points_counts_1000': self.data_path / 'proximity_points' / 'counts_1000',
            'skeleton': self.data_path / 'smoothed_skeletons'
        }

        self.files = {
            'connectome': self._list_files(self.dirs['connectome']),
            'contactome': self._list_files(self.dirs['contactome']),
            'proximity': self._list_files(self.dirs['proximity']),
            'proximity_points_counts': self._list_files(self.dirs['proximity_points_counts']),
            'proximity_points_counts_1000': self._list_files(self.dirs['proximity_points_counts_1000']),
            'skeleton': self._list_files(self.dirs['skeleton'], ext = '.json'),
        }

    def _list_files(self, directory_path, ext = '.csv'):
        """Loads all file paths from a specific Path object."""
        csv_files = {}
        if not directory_path.is_dir():
            print(f"Warning: Data directory '{directory_path}' not found.")
            return csv_files
        for filepath in sorted(directory_path.glob(f"*{ext}")):
            name = filepath.stem.split('_')[-1].split('.')[0]
            if self.rename_ds:
                name = rename_dataset_dict.get(name, name)
            csv_files[name] = filepath
        return csv_files
    
    def _filter_data_adj(self, df):
        """
        Filters names in the adjacency matrix

        Args:
            df (pd.DataFrame): The input adjacency matrix

        Returns:
            pd.DataFrame: A DataFrame containing only the filtered results.
        """
        base_types = ['sensory', 'inter', 'motor', 'modulatory']

        # Keep muscles if specified
        if self.include_muscle:
            base_types.append('muscle')
        mask = df.index.to_series().apply(lambda x: ntype(x) in base_types)

        # Keep postembryonic neurons if specified
        if not self.include_postemb:
            postemb_mask = ~df.index.to_series().apply(is_postemb)
            mask &= postemb_mask

        valid_indices = df.index[mask]
        valid_columns = df.columns.intersection(valid_indices)
        
        return df.loc[valid_indices, valid_columns]
    
    def _filter_prox_points(self, df):
        """
        Filters names in the proximity points matrix

        Args:
            df (pd.DataFrame): The input proximity points matrix

        Returns:
            pd.DataFrame: A DataFrame containing only the filtered results.
        """
        base_types = ['sensory', 'inter', 'motor', 'modulatory']

        # Keep muscles if specified
        if self.include_muscle:
            base_types.append('muscle')
        mask = df['neuron'].apply(lambda x: ntype(x) in base_types)

        # format specific to prox_points
        required_cols = ['neuron', 'x', 'y', 'z']
        filter_cols = [
            col for col in df.columns if col not in required_cols and (ntype(col) in base_types)
        ]

        # Keep postembryonic neurons if specified
        if not self.include_postemb:
            postemb_mask = ~df['neuron'].apply(is_postemb)
            mask &= postemb_mask
            filter_cols = [
                col for col in filter_cols if not is_postemb(col)
            ]

        valid_indices = df.index[mask]
        valid_columns = required_cols + filter_cols
        
        return df.loc[valid_indices, valid_columns]
    
    def adj_to_edgelist(self, df):
        """
        Converts an adjacency matrix to an DataFrame of edge information for each dataset.
        """
        df = self._filter_data_adj(df)
        df.reset_index(inplace=True)
        df.columns.values[0] = 'index'
        df = df.melt(id_vars="index")

        df["variable"] = df["variable"].replace(self.replacements)
        df["index"] = df["index"].replace(self.replacements)

        if self.npair_result:
            df["variable"] = df["variable"].apply(npair)
            df["index"] = df["index"].apply(npair)
            df = df.groupby(["variable","index"]).sum()
        else:
            df.set_index(["variable","index"], inplace = True)
        df.index = df.index.set_names(("pre","post"))
        return df
    
    def get_data_edgelist(self, datatype='connectome'):
        """
        Gets the combined edgelist for all datasets.
        """
        assert datatype in self.files.keys()

        csv_files = self.files[datatype]

        dfs = []
        for key, value in csv_files.items():
            df = pd.read_csv(value, index_col=0)
            df = self.adj_to_edgelist(df)
            df = df[df.value > 0]
            df = df.rename(columns={"value": key})
            dfs.append(df)

        G = pd.concat(dfs, axis=0).fillna(0)
        G = G.astype(int)
        G = G.groupby(level=[0,1]).sum()

        if datatype == 'contactome':
            index_set = set()
            for i in (G.index.tolist()):
                index_set.add(tuple(sorted(i)))
            index_list = list(index_set)
            G = G.loc[index_list]
        if datatype == 'proximity':
            merged_df = G.groupby(lambda idx_tuple: tuple(sorted(idx_tuple))).mean()
            merged_df.index = pd.MultiIndex.from_tuples(
                merged_df.index,
                names=df.index.names
            )
            return merged_df
        return G
    
    def get_data_adj(self, datatype='connectome'):
        """
        Gets a dictionary of adjacency matrices.
        """
        assert datatype in self.files.keys()

        csv_files = self.files[datatype]

        dfs = {}
        for key, value in csv_files.items():
            df = pd.read_csv(value, index_col=0)
            df = self._filter_data_adj(df)
            df.index = df.index.to_series().replace(self.replacements)
            df = df.rename(columns=self.replacements)

            if self.npair_result:
                df.index = df.index.to_series().apply(npair)
                df.columns = df.columns.to_series().apply(npair)
                df = df.groupby(df.index).sum()
                df = df.T.groupby(df.columns).sum().T
            dfs[key] = df
        return dfs
    
    def get_cable_lengths(self):
        """
        Loads cable lengths for all neurons.
        """
        skeleton_info = {}

        for f in self.dirs['skeleton']:
            match = re.search(r'^[^_]*', f.stem)
            with open(f, 'r') as file:
                data = json.load(file)
            skeleton_info[rename_dataset_dict[match.group(0)]] = data

        exclude_regex = ['exc']
        if not self.include_muscle:
            exclude_regex.append('BWM')

        cable_lengths = {}

        for dataset, data in skeleton_info.items():
            total_length = 0
            for neuron, properties in data.items():
                if any(re.search(r, neuron) for r in exclude_regex):
                    continue
                total_length += properties['length']
            cable_lengths[dataset] = total_length
        return pd.Series(cable_lengths)
    
    def get_volume(self):
        """
        Volume is a direct output from VAST as is therefore not calculated here

        Units are nm^3
        """
        volume = pd.Series({
            'L1-1': 140264774520,
            'L1-2': 219921688822,
            'L1-3': 204251501760,
            'L1-4': 206540239200,
            'L2': 547255631400,
            'L3': 564580826072,
            'adult-1': None,
            'adult-2': 1319942545440,
            'dauer-1': 1134361541000,
            'dauer-2': 874376648400,
            'dauer-3': None,
            'dauer-daf2': None,
        })
        return volume
    
    def get_proximity_points(self, datatype='counts', **kwargs):
        """
        Gets a dictionary of DataFrames representing proximity points.
        """
        assert datatype in ['counts', 'counts_1000']

        datasets = kwargs.get('datasets', [
            'L1-1', 'L1-2', 'L1-3', 'L1-4',
            'L2', 'L3', 'adult-2',
            'dauer-1', 'dauer-2'
        ])

        csv_files = self.files[f'proximity_points_{datatype}']

        dfs = {}
        
        for ds in datasets:
            assert ds in csv_files.keys()
            df = pd.read_csv(csv_files[ds])
            df = df.rename(columns=self.replacements)
            for k,v in self.replacements.items():
                df['neuron'] = df['neuron'].replace(k,v)
            if self.npair_result:
                required_cols = ['neuron', 'x', 'y', 'z']
                neuron_cols = [npair(col) for col in df.columns if col not in required_cols]
                df.columns = required_cols + neuron_cols
                df = df.T.groupby(df.columns).sum().T
                df['neuron'] = df['neuron'].apply(npair)
            dfs[ds] = self._filter_prox_points(df)
        return dfs
    
if __name__ == "__main__":
    data_manager = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=True,
        npair_result=False
    )
    connectome_edgelist = data_manager.get_data_edgelist(datatype='contactome')

    two_dauers = ['dauer-1','dauer-2']
    three_dauers = ['dauer-1','dauer-2','dauer-3']
    connectome_edgelist['two_dauers'] = connectome_edgelist.loc[:,two_dauers].astype(bool).sum(axis=1)==2
    connectome_edgelist['three_dauers'] = connectome_edgelist.loc[:,three_dauers].astype(bool).sum(axis=1)==3

    connectome_edgelist.to_csv('contactome_edgelist_ind.csv')

    # print(connectome_edgelist.head())

    # adj = data_manager.get_data_adj(datatype='proximity')
    # print(adj['adult-2'].head())

    # skeleton_lengths = data_manager.get_cable_lengths()
    # print(skeleton_lengths)