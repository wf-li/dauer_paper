import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import numpy as np
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

datasets = {
    'connectome': [
        'dataset1','dataset2','dataset3','dataset4',
        'dataset5','dataset6', 'dataset7', 'dataset8',
        'dauer-1','dauer-2','dauer-daf2'
        ],
    'contactome': [
        'dataset1','dataset2','dataset3','dataset4',
        'dataset5','dataset6', 'dataset8',
        'dauer-1','dauer-2','dauer-FIB'
        ],
    'nondauer': [
        'dataset1','dataset2','dataset3','dataset4',
        'dataset5','dataset6','dataset7','dataset8'
        ],
    'dauer': [
        'dauer-1','dauer-2','dauer-FIB','dauer-daf2'
        ]
}

def calculate_colors(values, cmap):
    colors = []
    for value in values:
        colors.append(cmap(value))
    return colors

def load_datasets(datatype, condition):
    return list(set(datasets[datatype]) & set(datasets[condition]))

def plot_adjacency_matrix(df, x_lines, y_lines, x_labels, y_labels, datatype,
                          cmap = 'Greys',
                          output_filename='heatmap.png',
                          vmax = None, vmin = 0):
    """
    Plots an adjacency matrix from a CSV file as a heatmap with specified dotted lines.
    """
    # set max and min values based on user specified threshold
    if not vmax:
        vmax = df.max().max()

    # Create the heatmap
    if datatype == 'connectome':
        plt.figure(figsize=(9.8, 11.3))
    elif datatype == 'contactome':
        plt.figure(figsize=(12,12))
    else:
        raise ValueError(f'Invalid datatype: {datatype}')
    ax = sns.heatmap(df, cmap=cmap, annot=False, xticklabels=False, yticklabels=False,
                    vmax = vmax, vmin = vmin, cbar=False, mask=(df==0))

    if x_lines:
        for line in x_lines:
            plt.axvline(line, color='red', linestyle='--', linewidth = 2)

    if y_lines:
        for line in y_lines:
            plt.axhline(line, color='red', linestyle='--', linewidth = 2)

    if x_labels:
        # Calculate the midpoints for the labels
        x_ticks = [0] + x_lines + [df.shape[1]]
        x_tick_labels_pos = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks)-1)]
        # Place the labels
        for i, label in enumerate(x_labels):
            plt.text(x_tick_labels_pos[i], -3, label,
                     fontsize = 14, ha='center', va='top')

    if y_labels:
        # Calculate the midpoints for the labels
        y_ticks = [0] + y_lines + [df.shape[0]]
        y_tick_labels_pos = [(y_ticks[i] + y_ticks[i+1]) / 2 for i in range(len(y_ticks)-1)]
        # Place the labels
        for i, label in enumerate(y_labels):
            # The rotation is set to 90 to make it more readable.
            plt.text(-0.5, y_tick_labels_pos[i], label,
                     fontsize = 12, ha='right', va='center', rotation=90)

    # Add titles and labels
    if datatype == 'connectome':
        plt.xlabel('Presynaptic', fontsize = 22, labelpad = 26)
        plt.ylabel('Postsynaptic', fontsize = 22, labelpad = 22)
        ax.xaxis.set_label_position('top')
    elif datatype == 'contactome':
        plt.xlabel(None)
        plt.ylabel(None)

    # Save the plot
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

# --- User-configurable section ---

# Specify the path to your CSV file
if __name__ == "__main__":
    datasets = ['data/processed/mean_connectome_dauer_intersection.csv']
    datatype = 'connectome'
    condition = 'nondauer'
    output_filename = 'dauer_intersection_with_colors.png'

    # datasets = load_datasets(datatype, condition)

    for dataset in datasets:
        # csv_file = f'data/raw/{datatype}s/{datatype}_{dataset}.csv' 
        csv_file = dataset
        # output_filename = f'figures/{datatype}_{dataset}_heatmap.png'
        # output_filename = f'{dataset}.png'

        # Specify the indices for the dotted lines
        xlines = [34, 58, 80]
        ylines = [34, 58, 80, 97]
        xlabels = ['Sensory','Inter','Motor', 'Modulatory']
        ylabels = ['Sensory','Inter','Motor', 'Modulatory', 'Muscle']
        vmin = -1
        vmax = 1
        
        data = pd.read_csv(csv_file, index_col = 0)

        if datatype == 'contactome':
            xlines.append(97)
            xlabels.append('Muscle')
            # data = np.log10(data[data>0])
            vmin = 4.5
            vmax = 8.5

        # xlabels = None
        # ylabels = None
        
        # cmap = LinearSegmentedColormap.from_list('halfgrey', ['#eeeeee','black'])

        # Generate the plot
        # plot_adjacency_matrix(data, xlines, ylines, xlabels, ylabels,
        #                         datatype = datatype, cmap = cmap,
        #                         output_filename = output_filename,
        #                         vmax = vmax, vmin = vmin)

classification = pd.read_csv('dauer_change_classification.csv', index_col = [0,1], names = ['cmap_names'])
remap = {0: 'Greys', 1: 'Oranges', -1: 'Blues'}
classification = classification.map(lambda x: remap[x])
vmaxes = {'Greys': 20, 'Oranges': 20, 'Blues': 20}
vmins = {'Greys': 0, 'Oranges': 0, 'Blues': 0}

def plot_connectome(data, colormap_classification, vmins, vmaxes,
                    x_lines, y_lines, x_labels, y_labels, output_filename = 'color_heatmap.png'):
    """
    Plots a connectome adjacency matrix with multiple colormaps.

    Args:
        data (pd.DataFrame): Adjacency matrix (postsynaptic x presynaptic).
        colormap_classification (pd.DataFrame): DataFrame mapping (pre, post) pairs to colormap names.
        vmins (dict): Dictionary mapping colormap names to their minimum values.
        vmaxes (dict): Dictionary mapping colormap names to their maximum values.
    """
    n_rows, n_cols = data.shape
    # 1. Initialize an RGBA image matrix with white for NaN cells
    # The shape is (rows, columns, 4 channels for RGBA)
    rgba_image = np.ones((n_rows, n_cols, 4))

    shared_cmap = LinearSegmentedColormap.from_list('halfgrey', ['#eeeeee','black'])
    dauer_cmap = LinearSegmentedColormap.from_list('dauers', ['#eeeeee','orange'])
    nondauer_cmap = LinearSegmentedColormap.from_list('nondauers', ['#eeeeee','blue'])

    # Pre-build colormap and normalization objects for efficiency
    colormaps = {'Greys': shared_cmap, 'Oranges': dauer_cmap, 'Blues': nondauer_cmap}
    normalizers = {name: mpl.colors.Normalize(vmin=vmins[name], vmax=vmaxes[name], clip = True) for name in vmaxes.keys()}

    # 2. Iterate through each cell of the adjacency matrix
    for i, post_neuron in enumerate(data.index):
        for j, pre_neuron in enumerate(data.columns):
            value = data.iloc[i, j]

            # Only process non-NaN cells
            if pd.notna(value):
                # 3. Validate if the connection exists in the classification
                connection = (pre_neuron, post_neuron)
                if connection not in colormap_classification.index:
                    raise ValueError(f"Connection from '{pre_neuron}' to '{post_neuron}' has value {value} but is not in the classification map.")

                # 4. Get the color for the cell
                cmap_name = colormap_classification.loc[connection].values[0]
                
                # Check if the colormap has defined limits
                if cmap_name not in colormaps.keys():
                    raise ValueError(f"Colormap '{cmap_name}' for connection {connection} does not have defined vmin/vmax values.")

                cmap = colormaps[cmap_name]
                norm = normalizers[cmap_name]
                
                # Assign the calculated RGBA color to the corresponding pixel
                rgba_image[i, j] = cmap(norm(value))

    # 5. Plot the final RGBA image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgba_image, interpolation='none', aspect='equal')
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])

    if x_lines:
        x_lines = [x-0.5 for x in x_lines]
        for line in x_lines:
            plt.axvline(line, color='red', linestyle='--', linewidth = 1)

    if y_lines:
        y_lines = [y-0.5 for y in y_lines]
        for line in y_lines:
            plt.axhline(line, color='red', linestyle='--', linewidth = 1)

    if x_labels:
        # Calculate the midpoints for the labels
        x_ticks = [0] + x_lines + [data.shape[1]]
        x_tick_labels_pos = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks)-1)]
        # Place the labels
        for i, label in enumerate(x_labels):
            plt.text(x_tick_labels_pos[i], -3, label,
                     fontsize = 12, ha='center', va='top')

    if y_labels:
        # Calculate the midpoints for the labels
        y_ticks = [0] + y_lines + [data.shape[0]]
        y_tick_labels_pos = [(y_ticks[i] + y_ticks[i+1]) / 2 for i in range(len(y_ticks)-1)]
        # Place the labels
        for i, label in enumerate(y_labels):
            # The rotation is set to 90 to make it more readable.
            plt.text(-0.5, y_tick_labels_pos[i], label,
                     fontsize = 12, ha='right', va='center', rotation=90)

    # Add titles and labels
    if datatype == 'connectome':
        plt.xlabel('Presynaptic', fontsize = 22, labelpad = 26)
        plt.ylabel('Postsynaptic', fontsize = 22, labelpad = 22)
        ax.xaxis.set_label_position('top')
    elif datatype == 'contactome':
        plt.xlabel(None)
        plt.ylabel(None)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()

plot_connectome(data, classification, vmins, vmaxes,
                xlines, ylines, xlabels, ylabels, output_filename)