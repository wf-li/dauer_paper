import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pca import pca_from_data
from adjustText import adjust_text

# --- Plotting ---
def plot_cumulative(sorted_abs_loadings, output_file = None, show = True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_abs_loadings, 'o-', label='Sorted Absolute Loadings', color='dodgerblue')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()

def plot_pca_component_eval(datatype, n_component_eval = 5, output_file = None, show = True):
    pca_eval, features_eval, _ = pca_from_data(datatype, n_components = n_component_eval)
    x = [i for i in range(1,n_component_eval+1)]
    plt.bar(x,pca_eval.explained_variance_ratio_)
    plt.xlabel('PCA Component')
    plt.ylabel('Explained Variance')
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()

def plot_circular_loadings(
        loadings, labels= None, n=None, p=None,
        separate=False, pc_x=0, pc_y=1, cmap = 'viridis',
        output_file = None, show = True):
    """
    Generates a circular plot of PCA loadings.

    Loadings are plotted as vectors from the origin. The function allows for selecting
    the most significant loadings based on multiple criteria.
    """
    # --- 1. Input Validation ---
    if n is None and p is None:
        raise ValueError("You must specify either 'n' (number) or 'p' (percentage) of loadings to plot.")
    if n is not None and p is not None:
        print("Warning: Both 'n' and 'p' were provided. 'n' will be used.")
        p = None

    num_features = loadings.shape[0]
    selected_indices = set()

    # --- 2. Selection Logic ---
    if separate:
        # Apply n or p separately to each PC
        pcs_to_consider = [pc_x, pc_y]
        for pc_index in pcs_to_consider:
            # Get the absolute values of loadings for the current PC
            pc_loadings = np.abs(loadings[:, pc_index])
            # Get the indices that would sort these loadings in descending order
            sorted_indices_pc = np.argsort(pc_loadings)[::-1]

            if n is not None:
                num_to_select = n
            else: # p is not None
                num_to_select = int(np.ceil(num_features * p))

            # Add the top indices to our set (sets automatically handle duplicates)
            selected_indices.update(sorted_indices_pc[:num_to_select])
    else:
        # Apply n or p to the combined magnitude (overall variance contribution)
        # Contribution is proportional to the squared distance from the origin
        magnitude = np.sum(loadings[:, [pc_x, pc_y]]**2, axis=1)
        sorted_indices_magnitude = np.argsort(magnitude)[::-1]

        if n is not None:
            num_to_select = n
        else: # p is not None
            num_to_select = int(np.ceil(num_features * p))

        selected_indices.update(sorted_indices_magnitude[:num_to_select])

    selected_indices = list(selected_indices)
    
    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal', adjustable='box')
    
    # Get the colormap
    colormap = plt.get_cmap(cmap)

    # Determine plot limits
    max_val = np.max(np.abs(loadings[selected_indices, :][:, [pc_x, pc_y]])) * 1.15
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add reference circle
    # The radius is set to the max magnitude of any *plotted* vector
    max_radius = np.sqrt(np.max(np.sum(loadings[selected_indices, :][:,[pc_x, pc_y]]**2, axis=1)))
    circle = patches.Circle((0, 0), max_radius, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(circle)

    # Add PC axes
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    # Plot vectors and labels
    texts = [] # for adjust_text
    for i in selected_indices:
        x = loadings[i, pc_x]
        y = loadings[i, pc_y]

        # Plot vector as an arrow
        angle = np.arctan2(np.abs(y), np.abs(x))
        # Normalize angle to the [0, 1] range for the colormap
        arrow_color = colormap(angle)

        # Plot vector as a colored arrow
        ax.arrow(0, 0, x, y, head_width=0.01 * max_val, head_length=0.02 * max_val, 
                 fc=arrow_color, ec=arrow_color)
        
        # Add label text
        if labels:
            label = labels[i]
            texts.append(ax.text(x * 1.05, y * 1.05, str(label),
                                ha='center', va='center', fontsize=9))
    if labels:
        adjust_text(texts, expand=(1.5,1.5))
        
    ax.set_xlabel(f"PC {pc_x + 1}", fontsize=12)
    ax.set_ylabel(f"PC {pc_y + 1}", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()

datatype = 'connectome'
pca, features, labels = pca_from_data(datatype)
output_file = "pca_loadings_1.png"
editable = False
n_components = 2

if editable:
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    output_file = output_file[:-4] + '.svg'

pca, features, edge_labels = pca_from_data(datatype, n_components, labels = 'edge_labels')
loadings = pca.components_.T 

# clean labels
labels_clean = [f'{pre}-{post}' for pre,post in edge_labels]
plot_circular_loadings(loadings, labels = None, p = 1, separate = True)

# # cumulative plot loading
# pc_loadings = loadings[:, 0]
# loadings_sorted = np.sort(np.abs(pc_loadings))[::-1]
# loadings_sorted = loadings_sorted[loadings_sorted > 0]

# xtickslabels = [f'PC{x}' for x in range(1, loadings.shape[1] + 1)]
# # ylabels = [f'{labels[i][0]}-{labels[i][1]}' for i in np.argsort(0-np.abs(loadings[:, 1]))]

# plt.figure(figsize=(10, 8))
# plt.imshow(loadings_sorted)
# plt.title('Feature Importance in PCA')
# plt.savefig('PCA_feat.svg')
# plt.show()