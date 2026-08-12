import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib import cm, colors
from collections import defaultdict
from adjustText import adjust_text

class Plotter:
    def __init__(self, output_path='results'):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created directory: {output_path}")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=12)
        plt.rc('axes', labelsize=10)
        self.linewidth = 1.5

    def _save_as(self, save_as, dpi=300, show=False):
        """Helper to save and show plots."""
        png_path = os.path.join(self.output_path, f'{save_as}.png')
        pdf_path = os.path.join(self.output_path, f'{save_as}.pdf')

        plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved figure to {png_path} and {pdf_path}")

        if show:
            plt.show()

    def _style_staged_x_axis(self, ax, fontsize=12, tick_pad=4):
        """Helper to apply common styling of larval stage x-axis."""
        ax.spines['bottom'].set_linewidth(self.linewidth)

        larval_stage_ends = [0, 16, 25, 34, 45]
        larval_stage_mids = [8, 20.5, 29.5, 39.5, 50]
        larval_stage_labels = ['L1', 'L2', 'L3', 'L4', 'Adult']

        ax.set_xlim([0, 55])
        ax.set_xticks(larval_stage_ends)
        ax.tick_params(axis='x', labelbottom=False)
        ax.set_xticks(larval_stage_mids, minor=True)
        ax.set_xticklabels(larval_stage_labels, minor=True, fontsize=fontsize)
        ax.tick_params(axis='x', which='minor', bottom=False, pad=tick_pad)

    def _style_dauer_x_axis(self, ax, tick_fontsize=12, label_fontsize=10, labelpad=2):
        """Helper to apply common styling of larval stage x-axis."""
        ax.set_xlim([0.5, 4.5])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['1', '2', '3', '4'], fontsize=tick_fontsize)
        ax.set_xlabel('Dauer', labelpad=labelpad, fontsize=label_fontsize)

    def plot_boxplot(self, data_df, groupings,
                    ylabel, ymin, ymax, yticks,
                    cmap_name='tab10',
                    group_gap=1.0,
                    box_width=0.8,
                    save_as='boxplot',
                    show_plot=False,
                    **kwargs):
        """
        Generates a vertical boxplot with grouped categories ("supercategories").
        
        Boxplots within the same supercategory are plotted next to each other
        and share a color, with extra spacing between supercategories.

        Args:
            data_df (pd.DataFrame): DataFrame where each column is a category
                                    to be plotted.
            groupings (dict): A dictionary mapping supercategory names (str)
                              to lists of column names (list of str).
                              Example:
                              {
                                 'Group A': ['col1', 'col2'],
                                 'Group B': ['col3', 'col4', 'col5']
                              }
            ylabel (str): Label for the Y-axis.
            ymin (float): Minimum value for the Y-axis.
            ymax (float): Maximum value for the Y-axis.
            yticks (list): List of tick positions for the Y-axis.
            save_as (str, optional): Filename (without extension). Defaults: 'boxplot'.
            cmap_name (str, optional): Matplotlib colormap name for coloring
                                       supercategories. Defaults: 'tab10'.
            group_gap (float, optional): The amount of extra spacing (in x-units)
                                         to add *between* supercategories. Defaults: 1.0.
            box_width (float, optional): Width of each individual box. Defaults: 0.8.
            show_plot (bool, optional): If True, calls plt.show(). Defaults: False.
            **kwargs: Additional keyword arguments passed to ax.boxplot().
                      Example: showfliers=False
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        try:
            cmap = cm.get_cmap(cmap_name)
            colors = [cmap(i) for i in range(len(groupings))]
        except ValueError:
            print(f"Warning: Colormap '{cmap_name}' not found. Defaulting to 'tab10'.")
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i) for i in range(len(groupings))]

        data_to_plot = []    # List of arrays, one for each box
        box_positions = []   # X-position for each box
        box_colors = []      # Color for each box
        xtick_labels = []    # Label for each supercategory
        xtick_positions = [] # X-position for each supercategory label

        current_pos = 0.5  # Start at 0.5 for centering
        
        # Iterate over the supercategories provided in the groupings dict
        for i, (group_name, columns) in enumerate(groupings.items()):
            color = colors[i % len(colors)]
            xtick_labels.append(group_name)
            
            group_start_pos = current_pos
            
            # Iterate over the individual columns within this supercategory
            for col_name in columns:
                if col_name not in data_df.columns:
                    print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping.")
                    continue
                
                # Append data (dropping NaNs)
                data_to_plot.append(data_df[col_name].dropna())
                box_positions.append(current_pos)
                box_colors.append(color)
                
                current_pos += 1  # Advance position for the next box
            
            # Calculate the center position for the supercategory label
            group_end_pos = current_pos - 1
            xtick_positions.append((group_start_pos + group_end_pos) / 2)
            
            # Add the extra gap *after* a group is plotted
            current_pos += group_gap

        # --- Plot the boxplots ---
        
        # Set default 'showfliers' to False if not provided
        if 'showfliers' not in kwargs:
            kwargs['showfliers'] = False

        bplot = ax.boxplot(
            data_to_plot,
            positions=box_positions,
            widths=box_width,
            patch_artist=True,  # Enable filling boxes with color
            **kwargs
        )

        # Color the boxes
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(self.linewidth / 2)

        # Style medians
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(self.linewidth)

        # Style whiskers and caps
        for partname in ('whiskers', 'caps'):
            for line in bplot[partname]:
                line.set_color('black')
                line.set_linewidth(self.linewidth / 2)

        # Style the Axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(self.linewidth)
        ax.spines['bottom'].set_linewidth(self.linewidth)
        ax.tick_params(width=self.linewidth, length=6, pad=1)

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, labelpad=2, fontsize=14)

        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=12)
        ax.tick_params(axis='x', length=0, pad=5)  # Hide tick marks, keep labels
        ax.set_xlim(left=0, right=current_pos - group_gap) 

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_xy_graph(self, x, y, x_dauer, y_dauer,
                        ylabel, ymin, ymax, yticks,
                        save_as='xy_graph', **kwargs):
        """
        Generates a two-panel plot for nondauer (staged) and dauer data.

        This creates a figure with two horizontally-aligned subplots (ax1, ax2)
        sharing a y-axis.
        1. Left (ax1): Displays nondauer data (x, y)
        2. Right (ax2): Displays dauer data (x_dauer, y_dauer)

        Args:
            x (array-like): X-values for nondauer data.
            y (array-like): Y-values for nondauer data.
            x_dauer (array-like): X-values for dauer data.
            y_dauer (array-like): Y-values for dauer data.
            ylabel (str): Label for the shared Y-axis.
            ymin (float): Minimum value for the Y-axis.
            ymax (float): Maximum value for the Y-axis.
            yticks (list): List of tick positions for the Y-axis.
            save_as (str, optional): Filename (without extension). Default: 'figure'.

            **kwargs:
                connect_points (bool, optional): If True, plots a line connecting
                    the mean y-value for each unique x-value in the nondauer plot.
                    Mutually exclusive with `best_fit_line`. Default: False.
                best_fit_line (bool, optional): If True, plots a linear regression
                    (best-fit) line for the nondauer data. Mutually exclusive
                    with `connect_points`. Default: False.
                dauer_lines (bool, optional): If True, draws horizontal/vertical
                    lines connecting dauer data points to the line on the
                    nondauer plot (requires `connect_points` or `best_fit_line`
                    to be True). Default: False.
                show_plot (bool, optional): Default: False.
        """
        dauer_lines = kwargs.get('dauer_lines', False)
        connect_points = kwargs.get('connect_points', False)
        best_fit_line = kwargs.get('best_fit_line', False)
        show_plot = kwargs.get('show_plot', False)

        if connect_points and best_fit_line:
            raise ValueError("connect_points and best_fit_line are mutually exclusive."
                             " Please set only one to True.")

        # Setup the figure and subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(5, 3),
            dpi=300,
            sharey=True,
            gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
        )

        # --- LEFT SUBPLOT (ax1): Timed Data Line Graph ---
        # Configure spines and ticks
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(self.linewidth)
        ax1.spines['bottom'].set_linewidth(self.linewidth)
        ax1.tick_params(width=self.linewidth, length=6, pad=1)
        ax1.set_ylim(ymin, ymax)
        ax1.set_yticks(yticks)
        ax1.set_ylabel(ylabel, labelpad=2, fontsize=14)

        # Add larval stage annotations
        self._style_staged_x_axis(ax1)

        # Plot the timed data points
        ax1.plot(
            x, y,
            color='k', marker='.',
            linewidth=0, markersize=20, label='_nolegend_'
        )
        
        ax1_line_xs = None
        ax1_line_ys = None
        ax1_line_func = None # Will store a function x = f(y)

        # Plot line connecting points (averaging y-values for duplicate x-values)
        if connect_points:
            unique_y = defaultdict(list)
            for i, val in enumerate(x):
                unique_y[val].append(y[i])
            xs_line = sorted(unique_y.keys())
            y_line = [np.mean(unique_y[val]) for val in xs_line]
            ax1.plot(xs_line, y_line, color='k', linewidth=4)
            # Store data for dauer_lines
            ax1_line_xs = xs_line
            ax1_line_ys = y_line

        elif best_fit_line:
            # Calculate linear regression: y = mx + b
            m, b = np.polyfit(x, y, 1)
            
            # Create x-values for the line spanning the plot's x-axis
            x_fit = np.array(ax1.get_xlim()) 
            y_fit = m * x_fit + b
            
            ax1.plot(
                x_fit, y_fit,
                color='k', linestyle='--',
                linewidth=1, label='Best Fit'
            )
            
            # Store the *inverse function* for dauer_lines: x = (y - b) / m
            if m != 0:
                ax1_line_func = lambda y_val: (y_val - b) / m
            else:
                # Handle horizontal line case (unlikely but safe)
                ax1_line_func = lambda y_val: ax1.get_xlim()[0]

        # --- RIGHT SUBPLOT (ax2) ---
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(self.linewidth)
        ax2.tick_params(axis='y', length=0) # Hide y-tick marks
        ax2.tick_params(axis='x', length=3)

        # Plot the dauer data
        ax2.plot(
            x_dauer, y_dauer,
            color='#1c75bc', marker='.',
            linestyle='None', markersize=20
        )

        self._style_dauer_x_axis(ax2)
        
        if dauer_lines: # Draw horizontal lines from dauer markersif dauer_lines: 
            if not connect_points and not best_fit_line:
                print("Warning: dauer_lines=True but no line specified on ax1 "
                      "(connect_points or best_fit_line). Skipping.")
            else:
                for xd, yd in zip(x_dauer, y_dauer):
                    start_x = 0 # Default intersection x-value

                    if connect_points and ax1_line_xs is not None:
                        # --- Logic for 'connect_points' intersection (as before) ---
                        intersection_x_coords = []
                        for i in range(len(ax1_line_xs) - 1):
                            y1, y2 = ax1_line_ys[i], ax1_line_ys[i+1]
                            x1, x2 = ax1_line_xs[i], ax1_line_xs[i+1]

                            if (y1 <= yd <= y2) or (y2 <= yd <= y1):
                                if y2 - y1 != 0:
                                    x_val = x1 + (x2 - x1) * (yd - y1) / (y2 - y1)
                                    if min(x1, x2) <= x_val <= max(x1, x2):
                                        intersection_x_coords.append(x_val)
                        if intersection_x_coords:
                            start_x = max(intersection_x_coords)

                    elif best_fit_line and ax1_line_func is not None:
                        # --- Logic for 'best_fit_line' intersection (new) ---
                        start_x = ax1_line_func(yd)
                        # Ensure the intersection is within the plot bounds
                        xlims = ax1.get_xlim()
                        start_x = np.clip(start_x, xlims[0], xlims[1])

                    # --- This plotting logic is now generic ---
                    if start_x >= ax1.get_xlim()[0]: # Only plot if intersection is valid
                        # Create a connection patch for the horizontal line
                        con = ConnectionPatch(xyA=(xd, yd), xyB=(start_x, yd),
                                              coordsA="data", coordsB="data",
                                              axesA=ax2, axesB=ax1,
                                              linestyle=":", color="#1c75bc", linewidth=1)
                        ax2.add_patch(con)

                        # Draw vertical line from intersection to x-axis
                        ax1.vlines(x=start_x, ymin=ymin, ymax=yd,
                                   colors='#1c75bc', linestyles=':', linewidth=1, zorder=0)

        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_multiple_xy(self, xs, ys, x_dauers, y_dauers,
                        ylabel, ymin, ymax, yticks,
                        save_as='multiple_xy', **kwargs):
        """
        Generates a figure with two subplots:
        1. Left: A line graph for nondauer data.
        2. Right: A marker plot for dauer data.
        """
        best_fit_line = kwargs.get('best_fit_line', False)
        show_plot = kwargs.get('show_plot', False)
        show_larval_stages = kwargs.get('show_larval_stages', False)
        colors = kwargs.get('colors')
        if colors is not None:
            assert type(colors) == list
        else:
            colors = ['k']*len(ys)
        
        # Setup the figure and subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(3.5, 6),
            dpi=300,
            sharey=True,
            gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
        )

        # --- LEFT SUBPLOT (ax1): Timed Data Line Graph ---

        # Configure spines and ticks
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(self.linewidth)
        ax1.spines['bottom'].set_linewidth(self.linewidth)
        ax1.tick_params(width=self.linewidth, length=6, pad=1)
        ax1.set_ylim(ymin, ymax)
        ax1.set_yticks(yticks)
        ax1.set_ylabel(ylabel, labelpad=2, fontsize=14)

        # Add larval stage annotations
        if show_larval_stages:
            self._style_staged_x_axis(ax1)

        # Plot the timed data points
        for i in range(0,len(ys)):
            ax1.scatter(
                xs, ys[i],
                color=colors[i], alpha = 0.75, 
                marker='.',edgecolor='black',
                linewidth=1, s=400
            )

            if best_fit_line:
                # Calculate linear regression: y = mx + b
                m, b = np.polyfit(xs, ys[i], 1)
                
                # Create x-values for the line spanning the plot's x-axis
                x_fit = np.array(ax1.get_xlim()) 
                y_fit = m * x_fit + b
                
                ax1.plot(
                    x_fit, y_fit,
                    color='k', linestyle='--',
                    linewidth=1, label='Best Fit'
                )

        # --- RIGHT SUBPLOT (ax2) ---
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(self.linewidth)
        ax2.tick_params(axis='y', length=0) # Hide y-tick marks
        ax2.tick_params(axis='x', length=3)

        # Plot the dauer data
        for i in range(0,len(y_dauers)):
            ax2.scatter(
                x_dauers, y_dauers[i],
                color=colors[i], alpha = 0.75,
                marker='.', edgecolor='black',
                linewidth=1, s=400
            )

        self._style_dauer_x_axis(ax2)

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_pca(self, xs, ys, xlabel, ylabel, labels,
                  save_as='pca', show_plot=False, **kwargs):
        """
        Generates a PCA plot.
        """
        clabels = kwargs.get('clabels', [0]*len(xs))
        cmap = kwargs.get('cmap', 'tab10')
        size = kwargs.get('size', 100)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(
            xs, ys,
            c=clabels, cmap=cmap, s=size
        )

        if labels:
            texts = []
            for i, txt in enumerate(labels):
                texts.append(
                    ax.text(xs[i], ys[i],
                    txt, ha = 'center', va = 'center', size = 8)
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.spines[['right','top']].set_visible(False)
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_pca_loadings_linear(
            self, loadings, feature_labels,
            n_features_to_show, plot_component,
            save_as='pca_loadings_linear', show_plot=True):
        """
        Generates a 1xN grid of cells
        """
        # --- Process Loadings ---
        pc_loadings = loadings[:, plot_component]
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]
        sorted_loadings = pc_loadings[sorted_indices]
        sorted_labels = feature_labels[sorted_indices]
        top_n_loadings = sorted_loadings[:n_features_to_show]
        top_n_labels = sorted_labels[:n_features_to_show]
        plot_data = top_n_loadings.reshape(1, -1)

        # --- Create Plot ---
        print(f"Generating plot for PC{plot_component + 1}...")
        plt.figure(figsize=(8, 2)) # Wide figure to accommodate labels

        # Center the diverging colormap (blue-white-red) at 0
        max_abs_val = np.max(np.abs(top_n_loadings))
        if max_abs_val == 0:
            max_abs_val = 1.0 # Avoid division by zero if all loadings are 0

        im = plt.imshow(plot_data, 
                        cmap='bwr',
                        aspect='auto',
                        vmin=-max_abs_val,
                        vmax=max_abs_val)

        plt.xticks(ticks=np.arange(len(top_n_labels)), 
                labels=top_n_labels, 
                rotation=90, 
                fontsize=8)
        # Hide y-axis (it's only 1 cell high)
        plt.yticks([])
        plt.colorbar(im, 
                    orientation='vertical', 
                    #  label='Loading Value', 
                    pad=0.01)
        plt.title(f'Top {n_features_to_show} PCA Loadings for PC{plot_component + 1} (Sorted by Absolute Value)')
        plt.xlabel('Features (Edge Labels)')
        plt.tight_layout()

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_pca_loadings_circular(
        self, loadings, labels=None, n=None, p=None,
        separate=False, pc_x=0, pc_y=1, pcs_to_consider = [0,1],
        cmap = 'Spectral',
        save_as = 'pca_loadings_circular', show_plot = True):
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
            if type(pcs_to_consider) == int:
                pcs_to_consider = [pcs_to_consider]
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

        # The reference circle radius is set to the max magnitude of any *plotted* vector
        max_radius = np.sqrt(np.max(np.sum(loadings[selected_indices, :][:,[pc_x, pc_y]]**2, axis=1)))
        circle = patches.Circle((0, 0), max_radius, edgecolor='gray', facecolor='none', linestyle='--')
        ax.add_patch(circle)

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

        texts = [] # for adjust_text
        for i in selected_indices:
            x = loadings[i, pc_x]
            y = loadings[i, pc_y]

            # Plot vector as an arrow
            angle = np.arctan2(np.abs(y), np.abs(x))
            # Normalize angle to the [0, 1] range for the colormap
            arrow_color = colormap(angle)
            ax.arrow(0, 0, x, y, head_width=0.01 * max_val, head_length=0.02 * max_val, 
                    fc=arrow_color, ec=arrow_color)
            
            if labels is not None:
                label = labels[i]
                texts.append(ax.text(x * 1.05, y * 1.05, str(label),
                                    ha='center', va='center', fontsize=9))
        if labels is not None:
            adjust_text(texts, expand=(1.2,1.5))
            
        ax.set_xlabel(f"PC {pc_x + 1}", fontsize=12)
        ax.set_ylabel(f"PC {pc_y + 1}", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.grid(False)

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_stacked_area(
                self, plot_data, 
                ylabel='Proportion', 
                ymin=0, ymax=1.0, 
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                colors=None,
                save_as='stacked_area_plot', **kwargs):
            """
            Generates a two-panel stacked area plot for staged and dauer data.

            The input DataFrame should have categories as rows and timepoints as columns.
            1. Left (ax1): Displays nondauer (staged) data.
            2. Right (ax2): Displays dauer data.

            Args:
                plot_data (pd.DataFrame): DataFrame with categories in rows and
                                        timepoints (e.g., 'L1-1', 'dauer-1') in columns.
                ylabel (str, optional): Label for the shared Y-axis. Defaults to 'Proportion'.
                ymin (float, optional): Y-axis minimum. Defaults to 0.
                ymax (float, optional): Y-axis maximum. Defaults to 1.0.
                yticks (list, optional): Y-axis tick positions. Defaults to [0, 0.2, ..., 1.0].
                colors (list, optional): List of colors for the categories. If None,
                                        uses the default 'tab10' colormap.
                save_as (str, optional): Filename (without extension). Defaults to 'stacked_area_plot'.
                
                **kwargs:
                    show_plot (bool, optional): If True, shows the plot. Default: False.
            """
            show_plot = kwargs.get('show_plot', False)

            # --- 1. Data Preparation ---
            # Define the hardcoded x-positions for staged and dauer timepoints
            staged_cols_map = {
                'L1-1': 0, 'L1-2': 5, 'L1-3': 8, 'L1-4': 16,
                'L2': 23, 'L3': 27, 'adult-1': 55, 'adult-2': 55
            }
            dauer_cols_map = {
                'dauer-1': 1, 'dauer-2': 2, 'dauer-daf2': 4
            }

            # Find which columns from our map are *actually* in the DataFrame
            staged_cols_present = [col for col in staged_cols_map if col in plot_data.columns]
            dauer_cols_present = [col for col in dauer_cols_map if col in plot_data.columns]

            # Sort the present columns by their x-position to ensure correct plotting order
            staged_cols_sorted = sorted(staged_cols_present, key=lambda col: staged_cols_map[col])
            dauer_cols_sorted = sorted(dauer_cols_present, key=lambda col: dauer_cols_map[col])

            # Get the corresponding x-values for the plot
            x_staged = [staged_cols_map[col] for col in staged_cols_sorted]
            x_dauer = [dauer_cols_map[col] for col in dauer_cols_sorted]

            # Get the data subsets
            data_staged = plot_data[staged_cols_sorted]
            data_dauer = plot_data[dauer_cols_sorted]
            
            # Get labels (categories) and set up colors
            labels = plot_data.index.tolist()
            n_categories = len(labels)

            if colors is None:
                # Use a default categorical colormap
                cmap = plt.get_cmap('tab10')
                plot_colors = [cmap(i) for i in range(n_categories)]
            else:
                if len(colors) < n_categories:
                    raise ValueError(f"Not enough colors. Need {n_categories}, got {len(colors)}")
                plot_colors = colors[:n_categories]

            # --- 2. Plot Setup ---
            # Setup the figure and subplots
            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=(5.5, 3.2),  # Slightly wider/taller to accommodate legend
                dpi=300,
                sharey=True,
                gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
            )

            # --- 3. LEFT SUBPLOT (ax1): Timed Data Stacked Area ---
            # Configure spines and ticks
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_linewidth(self.linewidth)
            ax1.spines['bottom'].set_linewidth(self.linewidth)
            ax1.tick_params(width=self.linewidth, length=6, pad=1)
            ax1.set_ylim(ymin, ymax)
            ax1.set_yticks(yticks)
            ax1.set_ylabel(ylabel, labelpad=2, fontsize=14)

            # Add larval stage annotations
            self._style_staged_x_axis(ax1)

            # Plot the staged stacked area
            # stackplot wants data as (categories, x-points)
            poly_collection = ax1.stackplot(
                x_staged,
                data_staged.values,
                labels=labels,
                colors=plot_colors,
                linewidth=0.25,
                edgecolor='white'
            )

            # --- 4. RIGHT SUBPLOT (ax2): Dauer Data Stacked Area ---

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_linewidth(self.linewidth)
            ax2.tick_params(axis='y', length=0)  # Hide y-tick marks
            ax2.tick_params(axis='x', length=3, width=self.linewidth)

            # Plot the dauer stacked area
            bottom_tracker = np.zeros(len(x_dauer))
            bar_width = 0.8  # Width of the bars
            for i in range(n_categories):
                category_data = data_dauer.values[i]
                color = plot_colors[i]
                
                ax2.bar(
                    x_dauer,
                    category_data,
                    bottom=bottom_tracker,
                    color=color,
                    width=bar_width,
                    linewidth=0.25,
                    edgecolor='white'
                )
                
                # Update the bottom for the next category
                bottom_tracker += category_data

            # Configure x-axis for dauer plot
            self._style_dauer_x_axis(ax2)

            # --- 5. Legend and Finalization ---
            # Add a shared legend above the plots
            fig.legend(
                poly_collection, labels,  # Use artists and labels from ax1
                loc='lower center',         # Anchor point of the legend
                bbox_to_anchor=(0.5, 0.95), # Position: (x=50% fig, y=95% fig)
                ncol=n_categories,          # All categories in one row
                frameon=False,
                fontsize=9
            )

            # Ensure ax1 (and its patches) are on top
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.patch.set_visible(False)  # Make ax1 transparent

            self._save_as(save_as, dpi=300, show=show_plot)
            plt.close(fig)

    def plot_violin_graph(
            self, data_nondauer, x_nondauer, data_dauer, x_dauer,
            ylabel, ymin, ymax, yticks, save_as='violin'):
        """
        Generates a figure with two subplots using violin plots.
        1. Left: Violins for nondauer data distributions vs. age.
        2. Right: Violins for dauer data distributions.
        
        Args:
            data_nondauer (list of arrays): List where each item is a 1D array
                                            of synapse counts for a nondauer dataset.
            x_nondauer (list): List of x-positions for nondauer violins.
            data_dauer (list of arrays): List where each item is a 1D array
                                         of synapse counts for a dauer dataset.
            x_dauer (list): List of x-positions for dauer violins.
            ylabel (str): Label for the shared Y-axis.
            ymin (float): Minimum value for the Y-axis.
            ymax (float): Maximum value for the Y-axis.
            yticks (list): List of tick positions for the Y-axis.
            save_as (str, optional): Filename (without extension). Defaults: 'plot'.
        """
        # Setup the figure and subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(5, 3),
            dpi=300,
            sharey=True,
            gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
        )
        
        dauer_color = '#90D5FF'

        # --- LEFT SUBPLOT (ax1): Timed Data Violin Plot ---

        # Configure spines and ticks
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(self.linewidth)
        ax1.spines['bottom'].set_linewidth(self.linewidth)
        ax1.tick_params(width=self.linewidth, length=6, pad=1)
        ax1.set_ylim(ymin, ymax)
        ax1.set_yticks(yticks)
        ax1.set_ylabel(ylabel, labelpad=2, fontsize=14)

        # Add larval stage annotations
        self._style_staged_x_axis(ax1)

        # Plot the nondauer violins
        parts_nondauer = ax1.violinplot(
            data_nondauer,
            positions=x_nondauer,
            widths=3.5,  # Adjust width to look good on the x-axis scale
            bw_method = 0.3,
            showmeans=False, showmedians=False, showextrema=True,
        )
        
        # Style the nondauer violins
        cmap = cm.get_cmap('viridis_r')
        norm = colors.Normalize(vmin=0, vmax=100)

        for i, pc in enumerate(parts_nondauer['bodies']):
            x_val = x_nondauer[i]
            color = cmap(norm(x_val))
            pc.set_facecolor(color)
            pc.set_edgecolor('k')
            pc.set_alpha(1)
            pc.set_linewidth(0.8)

        for partname in ('cbars','cmins','cmaxes'):
            vp = parts_nondauer[partname]
            vp.set_edgecolor('k')
            vp.set_linewidth(0.8)

        # Calculate and plot medians as scatter points
        medians_nondauer = [np.mean(d) for d in data_nondauer]
        ax1.scatter(x_nondauer, medians_nondauer, marker='o', color='k', s=15)

        # --- RIGHT SUBPLOT (ax2): Dauer Data Violin Plot ---
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(self.linewidth)
        ax2.tick_params(axis='y', length=0) # Hide y-tick marks
        ax2.tick_params(axis='x', length=3, width=self.linewidth)

        # Plot the dauer violins
        parts_dauer = ax2.violinplot(
            data_dauer,
            positions=x_dauer,
            widths=0.8,
            bw_method = 0.3,
            showmeans=False, showmedians=False, showextrema=True
        )

        # Style the dauer violins
        for pc in parts_dauer['bodies']:
            pc.set_facecolor(dauer_color)
            pc.set_edgecolor('k')
            pc.set_alpha(1)
            pc.set_linewidth(0.8)

        for partname in ('cbars','cmins','cmaxes'):
            vp = parts_dauer[partname]
            vp.set_edgecolor('k')
            vp.set_linewidth(0.8)

        # Calculate and plot medians as scatter points
        medians_dauer = [np.median(d) for d in data_dauer]
        ax2.scatter(x_dauer, medians_dauer, marker='o', color='k', s=15)

        # Configure x-axis for dauer plot
        self._style_dauer_x_axis(ax2)

        # --- Finalize and Save ---
        ax1.set_zorder(ax2.get_zorder()+1) # Ensure ax1 (and its patches) are on top
        ax1.patch.set_visible(False) # Make ax1 transparent

        self._save_as(save_as, dpi=300)