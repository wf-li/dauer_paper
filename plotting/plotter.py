import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import cm, colors
from collections import defaultdict

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

        # Plot the nondauer data points
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
        
        if dauer_lines: # Draw horizontal lines from dauer markers if dauer_lines: 
            if not connect_points and not best_fit_line:
                print("Warning: dauer_lines=True but no line specified on ax1 "
                      "(connect_points or best_fit_line). Skipping.")
            else:
                for xd, yd in zip(x_dauer, y_dauer):
                    start_x = 0 # Default intersection x-value

                    if connect_points and ax1_line_xs is not None:
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
                        start_x = ax1_line_func(yd)
                        # Ensure the intersection is within the plot bounds
                        xlims = ax1.get_xlim()
                        start_x = np.clip(start_x, xlims[0], xlims[1])

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

        self._save_as(save_as, dpi=300, show=show_plot)

    def plot_pca_loadings(
            self, loadings,
            bin_width=0.005,
            classes = None, class_order = None,
            save_as = 'pca_loadings', show_plot=False,
            **kwargs):
        """
        Plot individual loadings as a stacked dots in a
        histogram-like fashion.
        """
        default_color_map = {
            'dauer_increased': '#ed2024',
            'maintained': 'black',
            'dauer_decreased': '#abdbee',
            'variable': 'grey',
            'late_postembryonic': 'grey',
            'no_synapse': 'grey',
            'nan': 'grey'
        }
        color_map = kwargs.pop('color_map',default_color_map)
        norm = kwargs.pop('norm', None)
        figsize = kwargs.pop('figsize', (5.5,3))
        flip = kwargs.pop('flip', False)

        x_pos, y_pos, labels_out = stack_binned(
            loadings, bin_width=bin_width, classes=classes, 
            class_order=class_order,   # bottom -> top order in each column
        )
        if flip:
            x_pos = -x_pos

        plt.figure(figsize=figsize)
        if type(color_map) == dict:
            c = [color_map[c] for c in labels_out]
            plt.scatter(
                x_pos, y_pos, c=c,
                **kwargs
            )
        elif type(color_map) == colors.LinearSegmentedColormap:
            assert norm is not None # need norm for centered color
            plt.scatter(
                x_pos, y_pos,
                c=x_pos,
                cmap=color_map,
                norm=norm,
                **kwargs
            )
        ax = plt.gca()
        ax.spines[['top', 'right']].set_visible(False)

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

def stack_binned(x, bin_width, classes=None, class_order=None):
    """
    Bin 1D values into fixed-width x-columns and stack them into a
    histogram-like swarm. Within each column points are grouped by class
    (so colors stay contiguous) and fill from the bottom up (lowest free slot).

    Returns x_pos (bin centers), y_pos (0-based stack height), and the
    class label for each plotted point -- all aligned to each other.
    """
    x = np.asarray(x, dtype=float)
    classes_out = np.array([])

    lo = x.min()
    bin_idx = np.floor((x - lo) / bin_width).astype(int)
    bin_center = lo + (bin_idx + 0.5) * bin_width

    x_pos = np.empty_like(x)
    y_pos = np.empty_like(x)
    out = 0

    if classes is not None:
        classes = np.asarray(classes)
        if class_order is None:
            class_order = list(np.unique(classes))
        rank = {c: i for i, c in enumerate(class_order)}
        classes_out = np.empty(len(x), dtype=classes.dtype)

    for b in np.unique(bin_idx):
        members = np.where(bin_idx == b)[0]
        if classes is not None:
            members = sorted(members, key=lambda i: rank[classes[i]])
        for level, i in enumerate(members):
            x_pos[out] = bin_center[i]
            y_pos[out] = level
            if classes is not None:
                classes_out[out] = classes[i]
            out += 1
            
    return x_pos, y_pos, classes_out