import numpy as np
from sklearn.decomposition import PCA

def run_pca(edgetable, pca_components=2):
    features = np.array([edgetable[col].values for col in edgetable.columns])
    labels = edgetable.columns
    edge_labels = [f'{pre}-{post}' for pre,post in edgetable.index]

    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features)

    evr = pca.explained_variance_ratio_
    loadings = pca.components_.T

    result = {
        'features': features_pca.tolist(),
        'labels': labels.tolist(),
        'edge_labels': edge_labels,
        'explained_variance': evr.tolist(),
        'loadings': loadings.tolist()
    }
    return result
