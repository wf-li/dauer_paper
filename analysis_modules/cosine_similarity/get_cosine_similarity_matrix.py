from pathlib import Path
import numpy as np
import networkx as nx
from analysis_modules.src.data_manager import DataManager
from analysis_modules.src.utils import exclude_postemb_edgetable

def calculate_cosine_similarity(graphs: np.ndarray):
    num_graphs = len(graphs)
    if num_graphs == 0:
        return np.array([])

    similarity_matrix = np.zeros((num_graphs, num_graphs))

    # Ensure adjacency matrices have same nodes
    all_nodes = set()
    for graph in graphs:
        all_nodes.update(graph.nodes())
    node_list = sorted(list(all_nodes))

    adj_matrices = [
        nx.adjacency_matrix(g, nodelist=node_list, weight='weight').toarray()
        for g in graphs
    ]
    # Calculate pairwise cosine similarity
    for i in range(num_graphs):
        for j in range(i, num_graphs):
            vec1 = adj_matrices[i].flatten()
            vec2 = adj_matrices[j].flatten()

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                raise ValueError('At least one of the matrices are empty.')

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix

output_path = Path('analysis_modules/cosine_similarity/outputs')

for datatype in ['connectome','proximity','drive']:
    output_file = Path(f'{datatype}_similarity_matrix.json')

    dm = DataManager(
        data_path = 'data',
        include_postemb=True,
        include_muscle=True,
        npair_result=True
    )

    dataset_selection = {
        'connectome': [
            'L1-1','L1-2','L1-3','L1-4',
            'L2','L3','adult-1','adult-2',
            'dauer-1','dauer-2','dauer-daf2'
        ],
        'proximity': [
            'L1-1','L1-2','L1-3','L1-4',
            'L2','L3','adult-2',
            'dauer-1','dauer-2','dauer-3'
        ],
        'drive': [
            'L1-1','L1-2','L1-3','L1-4',
            'L2','L3','adult-2',
            'dauer-1','dauer-2'
        ],
    }
    dataset_labels = {
        'connectome': [
            'L1','L1','L1','L1',
            'L2','L3','A-1','A-2',
            'D-1','D-2','D-4'
        ],
        'proximity': [
            'L1','L1','L1','L1',
            'L2','L3','A-2',
            'D-1','D-2','D-3'
        ],
        'drive': [
            'L1','L1','L1','L1',
            'L2','L3','A-2',
            'D-1','D-2'
        ],
    }

    datasets = dataset_selection[datatype]

    if datatype == 'drive':
        edgetable = dm.get_drive_edgetable(consistent=False)
    else:
        edgetable = dm.get_data_edgetable(datatype)
    edgetable = edgetable.loc[:,datasets]

    if datatype == 'connectome':
        edgetable = np.sqrt(edgetable)
    if datatype == 'proximity':
        edgetable[edgetable<10]=0
        edgetable = edgetable.replace(0,np.nan)
        edgetable = np.log(edgetable)

    # remove PVW PLN HSN
    to_remove = ['PVW','PLN','HSN']
    el = exclude_postemb_edgetable(edgetable, datasets, to_remove)
    el = el.loc[el.loc[:,datasets].sum(axis=1)>0, datasets]
    el = np.sqrt(el)

    graphs = []
    labels = dataset_labels[datatype]

    for col in edgetable.columns:
        graph = nx.DiGraph()
        graph.add_nodes_from(
            edgetable.index.get_level_values(0).unique(
                ).union(edgetable.index.get_level_values(1).unique())
        )

        for index,value in edgetable[col][edgetable[col]>0].items():
            graph.add_edge(index[0], index[1], weight=value)

        graphs.append(graph)

    similarity_matrix = calculate_cosine_similarity(graphs)
    output = {
        'matrix': similarity_matrix.tolist(),
        'labels': labels
    }

    import json
    with open(output_path / output_file, 'w+') as fp:
        json.dump(output, fp, indent=4)