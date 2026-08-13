from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.src.neuron_info import ntype, dark_colors
from analysis_modules.src.data_manager import DataManager
import pandas as pd

# parameters
dataset = 'dauer-1'
data_type = 'connectome' # 'connectome', 'proximity'

output_name = Path(f'cytoscape_graph_{dataset}.png')

node_info_path = 'plotting/utils/node_coordinates.csv'
node_info = pd.read_csv(node_info_path)
node_info.index = node_info['name']

dm = DataManager(
    data_path = 'data',
    include_muscle = True,
    include_postemb = True,
    npair_result = True
)
data = dm.get_data_edgetable(data_type)
plot_data = data[dataset].rename('weight')

# data params
ignore_neurons = ['PVW','PLN','HSN']
minimum_edge_weight = 2
ignore_self_edges = True

# plotting params
n_size_min,n_size_max = 50,300.0
e_width_min,e_width_max = 0.25,2.5
e_color_min,e_color_max = 0.8,0.0

# ignore neurons
plot_data = plot_data[~plot_data.index.get_level_values(0).isin(ignore_neurons)]
plot_data = plot_data[~plot_data.index.get_level_values(1).isin(ignore_neurons)]

# ignore self edge loop
if ignore_self_edges:
    plot_data = plot_data[~(plot_data.index.get_level_values(0) == plot_data.index.get_level_values(1))]

# set edge weight minimum
plot_data = plot_data[plot_data >= minimum_edge_weight]

outputs = plot_data.astype(bool).groupby(level=0).sum()
inputs = plot_data.astype(bool).groupby(level=1).sum()
node_degree = (outputs+inputs).fillna(0)

G = nx.DiGraph()
pos = {
    key: (value['x']*0.8, -value['z'])
    for key, value in node_info[['x','z']].T.to_dict().items()
}

# formulas for variable node size / edge width / edge color
n_min, n_max = node_degree.min(), node_degree.max()
e_min, e_max = plot_data.min(), plot_data.max()

def _norm(w, lo, hi, gamma=1.0):
    return float(np.clip((w - lo) / (hi - lo), 0.0, 1.0)) ** gamma

def node_size_formula(w):
    t = _norm(w, n_min, n_max, gamma=1)
    r0, r1 = np.sqrt(n_size_min), np.sqrt(n_size_max)
    return (r0 + t * (r1 - r0)) ** 2

def edge_size_formula(w):
    t = _norm(w, e_min, e_max, gamma=0.7)
    return e_width_min + t * (e_width_max - e_width_min)

def edge_color_formula(w):
    t = _norm(w, e_min, e_max, gamma=0.7)
    g = e_color_min + t * (e_color_max - e_color_min)
    return (g, g, g, 1-g)

nodelist = list(node_degree.index)
node_sizes = np.array([node_size_formula(w) for w in node_degree[nodelist]])
node_colors = [dark_colors[ntype(n)] for n in nodelist]

# plotting
fig, ax = plt.subplots(figsize=(6, 6))
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=nodelist,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors = 'White',
    linewidths = 0.5
).set_zorder(2)

nx.draw_networkx_edges(
    G,
    pos,
    nodelist=nodelist,
    node_size=node_sizes,
    edgelist = plot_data.index,
    width=[edge_size_formula(w) for w in plot_data],
    edge_color=[edge_color_formula(w) for w in plot_data],
    arrows=True,
    arrowsize=5,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.05",
)

plt.axis('off')
plt.tight_layout()

output_path = Path('figures')
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {output_path}")
plt.savefig(output_path / output_name, dpi=300, bbox_inches='tight')