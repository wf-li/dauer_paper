import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import log10

def plot_graph(G, datatype, output_path = './'):
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

    pos = nx.get_node_attributes(G, 'pos')
    z_orders = nx.get_node_attributes(G, 'z_order')
    node_color_map = nx.get_node_attributes(G, 'color')

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
    
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    plt.savefig(f'{output_path}')

if __name__ == "__main__":
    datatype = 'connectome' # 'connectome' or 'contactome'

    assert datatype in ['connectome', 'contactome']

    import pandas as pd
    from analysis_modules.src.data_manager import DataManager
    from analysis_modules.cytoscape_graph.make_nx_graph import make_graph
    
    node_metadata_path = 'analysis_modules/cytoscape_graph/node_metadata.csv'

    replacements = {
        "PVWL_or_R_1": "PVWL",
        "PVWL_or_R_2": "PVWL",
        "PVWL_or_R_3": "PVWR",
        "RICRa": "RICR",
        "RICRp": "RICR"
    }

    data = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=True,
        npair_result=False,
        replacements=replacements
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
    if datatype == 'connectome':
        datasets.extend(['adult-1', 'dauer-daf2'])
    elif datatype == 'contactome':
        datasets.append('dauer-3')

    for dataset in datasets:
        output_path = f'figures/cytoscape_graphs/{datatype}_graph/{dataset}_graph.png'
        G = make_graph(node_df, edge_df, dataset = dataset, filter_nodes = filter_nodes)
        plot_graph(G, datatype, output_path=output_path)