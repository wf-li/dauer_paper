from pathlib import Path
import numpy as np
import pandas as pd
import json

dauer_label = {
    'L1-1': 'nondauer',
    'L1-2': 'nondauer',
    'L1-3': 'nondauer',
    'L1-4': 'nondauer',
    'L2': 'nondauer',
    'L3': 'nondauer',
    'adult-1': 'nondauer',
    'adult-2': 'nondauer',
    'dauer-1': 'dauer',
    'dauer-2': 'dauer',
    'dauer-3': 'dauer',
    'dauer-daf2': 'dauer'
}

def get_adjusted_loadings(pca_data: dict, sorted=True) -> pd.DataFrame:
    """
        PCA data obtained from output of pca.run_pca()
        Can also be loaded from .json from output_pca_json

        Outputs PC1, PC2 and centroid axis loading values
    """
    pca_scores = np.array(pca_data['features'])
    labels = np.array([dauer_label[label] for label in pca_data['labels']])
    edge_labels = np.array([[edge.split('-')[0],edge.split('-')[1]] for edge in pca_data['edge_labels']])
    edge_pre = edge_labels[:,0]
    edge_post = edge_labels[:,1]
    loadings = np.array(pca_data['loadings'])

    # Find Cluster Centroids in 2D PCA space
    coords_nondauer = pca_scores[labels == 'nondauer']
    coords_dauer = pca_scores[labels == 'dauer']
    centroid_nondauer = np.mean(coords_nondauer, axis=0)
    centroid_dauer = np.mean(coords_dauer, axis=0)

    # Calculate the Normalized Direction Vector (Axis of Separation)
    direction_vector = centroid_nondauer - centroid_dauer
    norm_vector = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / norm_vector

    # Project original loadings onto the new axis
    # Matrix multiplication: (k, 2) @ (2,) -> (k,)
    new_axis_loadings = np.dot(loadings, unit_vector)

    results_df = pd.DataFrame({
        'pre': edge_pre,
        'post': edge_post,
        'PC1_loading': loadings[:, 0],
        'PC2_loading': loadings[:, 1],
        'separation_axis_loading': new_axis_loadings
    })
    results_df.set_index(['pre','post'], inplace=True)
    if sorted:
        results_df = results_df.sort_values(by='separation_axis_loading', ascending=False)

    print("Top 5 edges driving towards nondauer:")
    print(results_df.head())
    print("\nTop 5 edges driving towards dauer:")
    print(results_df.tail())
    return results_df

if __name__ == "__main__":
    input_path = Path('pca_drive.json')

    with open(input_path, 'r') as f:
        pca_data = json.load(f)

    results_df = get_adjusted_loadings(pca_data, sorted=True)

    results_df.to_csv(input_path / "results_df.csv")