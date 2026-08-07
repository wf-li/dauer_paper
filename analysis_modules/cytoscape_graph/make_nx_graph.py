import pandas as pd
import networkx as nx
import matplotlib.patches as mpatches

dark_colors = {
    'sensory': '#fda0fd',
    'inter': '#ff442f',
    'motor': '#5cafff',
    'modulatory': '#ffc000',
    'muscle': '#7ec95a',
    'other': '#d9d9d9',
    'nonvalid': '#000000',
}

def make_graph(node_df, edge_df, dataset, filter_nodes = []):
    """
    Loads graph data from CSV files and plots the directed graph.

    Args:
        node_file (str): Path to the CSV file with node metadata (name, type, x, y, zorder).
        edge_file (str): Path to the CSV file with edge data.
    """
    G = nx.DiGraph()

    # Add nodes and their attributes from the node metadata file
    for _, row in node_df.iterrows():
        node_name = row['name']
        if node_name in filter_nodes:
            continue
        G.add_node(
            node_name,
            type=row['type'],
            pos = (row['x'], row['y']),
            color = dark_colors.get(row['type'], '#808080'),
            z_order = row['zorder']
        )

    # Add edges
    for edge, row in edge_df.iterrows():
        weight = row[dataset]
        if weight == 0:
            continue
        source = edge[0]
        target = edge[1]
        # Ensure both source and target nodes exist before adding an edge
        if source in G and target in G:
            # if weight < 2:
            #     continue
            G.add_edge(source, target, weight = weight)
        else:
            print(f"Warning: Skipping edge '{source}' -> '{target}' because one or both nodes are not in the metadata.")
            pass
    return G

if __name__ == "__main__":
    datatype = 'connectome'

    from ..src.data_manager import DataManager
    
    node_metadata_path = 'analysis_modules/cytoscape_graph/node_metadata.csv'

    data = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=True,
        npair_result=False
    )

    node_df = pd.read_csv(node_metadata_path)
    edge_df = data.get_data_edgelist(type=datatype)

    filter_nodes = [
        'GLRR','GLRL','GLRDL','GLRDR','GLRVL','GLRVR',
        'CEPshVR','CEPshVL','CEPshDL','CEPshDR',
    ]

    datasets = [
        'L2','L3','adult-2','dauer-1','dauer-2'
    ]

    G = make_graph(node_df, edge_df, dataset = datasets[0], filter_nodes = filter_nodes)
    print(G.nodes['ADAL'])