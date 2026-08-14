import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ..src.neuron_info import ntype, npair, is_postemb

column_order = [
    'L1-1','L1-2','L1-3','L1-4','L2','L3',
    'adult-1','adult-2','dauer-1','dauer-2','dauer-3','dauer-daf2'
]
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
        """
        self.data_path = Path(data_path)
        self.include_postemb = kwargs.get('include_postemb', True)
        self.include_muscle = kwargs.get('include_muscle', False)
        self.npair_result = kwargs.get('npair_result', False)
        self.replacements = kwargs.get('replacements', {})

        self.dirs = {
            'connectome': self.data_path / 'connectomes' / 'individual_cells',
            'contactome': self.data_path / 'contactomes' / 'individual_cells',
            'proximity': self.data_path / 'proximity',
            'skeleton': self.data_path / 'smoothed_skeletons'
        }

        self.files = {
            'connectome': self._list_files(self.dirs['connectome']),
            'contactome': self._list_files(self.dirs['contactome']),
            'proximity': self._list_files(self.dirs['proximity']),
            'skeleton': self._list_files(self.dirs['skeleton'], ext = '.json'),
        }

    def _list_files(self, directory_path, ext = '.csv'):
        """Loads all file paths from a specific Path object."""
        files = {}
        if not directory_path.is_dir():
            print(f"Warning: Data directory '{directory_path}' not found.")
            return files
        for filepath in sorted(directory_path.glob(f"*{ext}")):
            name = filepath.stem.split('_')[-1].split('.')[0]
            files[name] = filepath
        return files
    
    def _filter_data_adj(self, df):
        """
        Filters names in data formatted as adjacency matrices

        Args:
            df (pd.DataFrame): The input adjacency matrix

        Returns:
            pd.DataFrame: A DataFrame containing only the filtered results.
        """
        base_types = ['sensory', 'inter', 'motor', 'modulatory']

        # Keep muscles
        if self.include_muscle:
            base_types.append('muscle')
        mask = df.index.to_series().apply(lambda x: ntype(x) in base_types)

        # Keep postembryonic neurons
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
    
    def adj_to_edgetable(self, df):
        """
        Converts an adjacency matrix to an DataFrame of edge information for each dataset.
        """
        df = self._filter_data_adj(df)
        df = (
            df.rename_axis(index="index", columns="variable")
            .stack()
            .rename("value")
            .reset_index()
        )

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
    
    def get_data_edgetable(self, datatype='connectome'):
        """
        Gets the combined edgetable for all datasets.
        """
        assert datatype in self.files.keys()

        csv_files = self.files[datatype]

        dfs = []
        for key, value in csv_files.items():
            df = pd.read_csv(value, index_col=0)
            df = self.adj_to_edgetable(df)
            df = df[df.value > 0]
            df = df.rename(columns={"value": key})
            dfs.append(df)

        G = pd.concat(dfs, axis=0).fillna(0)
        G = G.astype(int)
        G = G.groupby(level=[0,1]).sum()
        ordered_cols = [col for col in column_order if col in G.columns]
        G = G[ordered_cols]

        # remove duplicates in symmetrical matrices
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

    def get_drive_edgetable(
            self, ds=['L2','L3','adult-2','dauer-1','dauer-2'],
            consistent=True):
        """
        Gets the synapse drive for all datasets or
        selected datasets ds if consistent is True.
        """
        con_el = self.get_data_edgetable('connectome')
        con_el = np.sqrt(con_el)
        prox_el = self.get_data_edgetable('proximity')

        # filter proximity < 10 points
        prox_el[prox_el<10]=0
        prox_el_invert = prox_el.swaplevel('pre', 'post')
        prox_el_invert.index.names = ['pre', 'post']
        prox_el_invert = prox_el_invert.query("pre != post")
        prox_undirected = pd.concat([prox_el, prox_el_invert]).sort_index()
        prox_el = prox_undirected.groupby(['pre', 'post']).sum()
        prox_el = prox_el.replace(0,np.nan)
        prox_el = np.log(prox_el)
        drive = con_el / prox_el
        if consistent: # proximity value > 0 in all datasets
            contact_rule = drive[drive[ds].isna().sum(axis=1)==0].index
            drive = drive.loc[drive.index.isin(contact_rule),ds]
        return drive
    
    def get_cable_lengths(self):
        """
        Loads cable lengths for all neurons.
        """
        skeleton_info = {}

        for f in self.dirs['skeleton']:
            match = re.search(r'^[^_]*', f.stem)
            with open(f, 'r') as file:
                data = json.load(file)

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
        Volume calculated from segmentation voxels

        Units are nm^3
        """
        volume = pd.Series({
            'L1-1': 84080778240,
            'L1-2': 148189059480,
            'L1-3': 136209323520,
            'L1-4': 139844505600,
            'L2': 355876392960,
            'L3': 408271907190,
            'adult-1': None,
            'adult-2': 1033025495040,
            'dauer-1': 551742182400,
            'dauer-2': 554858521600,
            'dauer-3': 326481722000,
            'dauer-daf2': None,
        })
        return volume

    def specify_proximity_points_dir(self, directory_path, ext = '.csv'):
        """Proximity point matrices are downloaded separately for speed and convenience."""
        self.dirs['proximity_points'] = Path(directory_path)
        self.files['proximity_points'] = self._list_files(self, directory_path, ext)
        print(f"Set proximity points directory as {directory_path}.")

    def get_proximity_points(self, **kwargs):
        """
        Gets a dictionary of DataFrames representing proximity points.
        """
        datasets = kwargs.get('datasets', ['L1-1'])
        if type(datasets) == str:
            datasets = list(datasets)

        csv_files = self.files['proximity_points']

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