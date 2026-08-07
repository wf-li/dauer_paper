import pandas as pd
import networkx as nx
from neuron_info import dark_colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import log10

def G_from_csv(node_df, edge_df, dataset, filter_nodes = []):
    """
    Loads graph data from CSV files and plots the directed graph.

    Args:
        node_file (str): Path to the CSV file with node metadata (name, type, x, y, zorder).
        edge_file (str): Path to the CSV file with edge data.
        colors (dict): A dictionary mapping node types to their colors.
    """
    G = nx.DiGraph()

    # Add nodes and their attributes from the node metadata file
    pos = {}
    node_color_map = {}
    z_orders = {}

    for _, row in node_df.iterrows():
        node_name = row['name']
        if node_name in filter_nodes:
            continue
        G.add_node(node_name, type=row['type'])
        pos[node_name] = (row['x'], row['y'])
        node_color_map[node_name] = dark_colors.get(row['type'], '#808080') # Default to gray if type not in colors
        z_orders[node_name] = row['zorder']

    # Add edges from the edge data file
    for _, row in edge_df.iterrows():
        weight = row[dataset]
        if weight == 0:
            continue
        source = row['pre']
        target = row['post']
        # Ensure both source and target nodes exist before adding an edge
        if source in G and target in G:
            # if weight < 2:
            #     continue
            G.add_edge(source, target, weight = weight)
        else:
            print(f"Warning: Skipping edge '{source}' -> '{target}' because one or both nodes are not in the metadata.")
            pass
    return G, pos, z_orders, node_color_map

def plot_graph(G,pos,z_orders,node_color_map,datatype,output_path = './'):
    fig, ax = plt.subplots(figsize=(10,10))

    
    alphas = []
    edge_weights = []
    for u,v in G.edges():
        weight_uv = G[u][v]['weight']
        if datatype == 'contactome':
            try:
                weight_uv = log10(weight_uv)
                if ((weight_uv-5)/5) > 0.6:
                    alphas.append(0.6)
                elif ((weight_uv-5)) < 0.1:
                    alphas.append(0.1)
                else:
                    alphas.append((weight_uv-5)/5)
            except:
                raise ValueError(f'{weight_uv} is not a valid weight')
        if weight_uv > 20:
            edge_weights.append(5)
        else:
            try:
                edge_weights.append(weight_uv*0.25)
            except:
                raise ValueError(f'Edge {u} to edge {v} value error: {G[u][v]['weight']}')

    # Draw the edges first so they appear behind the nodes
    edge_alpha = 0.4
    if datatype == 'contactome':
        edge_alpha = alphas
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=edge_alpha,
        edge_color='black',
        width=edge_weights,
        arrows=False,
        # arrowstyle='-|>',
        # arrowsize=6,
        node_size=300,
    )

    # Sort nodes by their z-order to draw them in the correct sequence (higher z-order on top)
    sorted_nodes = sorted(list(G.nodes()), key=lambda n: z_orders.get(n, 0))

    # Draw each node individually to respect the z-order
    for node in sorted_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            ax=ax,
            node_size=300,
            node_color=node_color_map.get(node, '#808080'),
            edgecolors = 'white'
        )

    # Create a custom legend for the node types
    # legend_patches = [mpatches.Patch(color=color, label=label.capitalize()) for label, color in dark_colors.items()]
    # ax.legend(handles=legend_patches, loc='upper right', fontsize=12, title='Node Types', title_fontsize=14, frameon=True)

    # Customize the plot appearance
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    plt.savefig(f'{output_path}')
    
    # Show the plot
    # plt.show()

if __name__ == '__main__':
    datatype = 'contactome' # 'connectome' or 'contactome'
    node_file = 'node_metadata.csv'
    edge_file = f'all_{datatype}_edges.csv'
    ftype = 'png' # output file type

    # datasets = [
    #     'L2','L3','adult_sem','dauer_sem_1','dauer_sem_2','dauer_FIB'
    # ]
    datasets = ['dauer_daf-2']

    # dataset = 'adult_sem'
    fname_mapping = {
        'L2': 'dataset5',
        'L3': 'dataset6',
        'adult_tem': 'dataset7',
        'adult_sem': 'dataset8',
        'dauer_sem_1': 'dauer-1',
        'dauer_sem_2': 'dauer-2',
        'dauer_FIB': 'dauer-3',
        'dauer_daf2': 'dauer_daf-2'
    }
    
    for dataset in datasets:
        output_path = f'./{datatype}_graph_figures/{fname_mapping[dataset]}_graph.{ftype}'

        filter_nodes = [
            'GLRR','GLRL','GLRDL','GLRDR','GLRVL','GLRVR',
            'CEPshVR','CEPshVL','CEPshDL','CEPshDR',
        ]

        try:
            # Load node and edge data from CSV files
            node_df = pd.read_csv(node_file)
            edge_df = pd.read_csv(edge_file)
        except FileNotFoundError as e:
            print(f"Error: {e}. Please make sure the CSV files are in the correct directory.")

        # # Call the function to generate and display the plot
        G, pos, z_orders, node_color_map = G_from_csv(node_df, edge_df, dataset, filter_nodes)
        plot_graph(G, pos, z_orders, node_color_map, datatype, output_path)