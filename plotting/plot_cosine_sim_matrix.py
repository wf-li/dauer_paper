import numpy as np
from pathlib import Path
from plotter import Plotter
import json

datatype = 'connectome'

matrix_path = Path('analysis_modules/cosine_similarity/outputs')
matrix_file = Path(f'{datatype}_similarity_matrix.json')

with open(matrix_path / matrix_file) as f:
    data = json.load(f)

similarity_matrix = np.array(data['matrix'])
labels = data['labels']

figure_generator = Plotter(output_path = 'figures/cosine_sim')
figure_generator.plot_cosine_similarity(
    similarity_matrix, labels=labels,
    annot=False, cmap='Blues', vmax=1, vmin=0.6,
    save_as = f'cosine_similarity_{datatype}'
)