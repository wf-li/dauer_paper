import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

def get_kmeans_labels(adjs, n_clusters=2, random_state=47):
    similarity_matrix = cosine_similarity(adjs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    return cluster_labels

def run_pca(edgetable, pca_components=2, kmeans_clusters=2):
    features = np.array([edgetable[col].values for col in edgetable.columns])
    labels = edgetable.columns
    edge_labels = [f'{pre}-{post}' for pre,post in edgetable.index]

    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features)

    evr = pca.explained_variance_ratio_
    loadings = pca.components_.T
    cluster_labels = get_kmeans_labels(features, kmeans_clusters)

    result = {
        'features': features_pca.tolist(),
        'labels': labels.tolist(),
        'edge_labels': edge_labels,
        'explained_variance': evr.tolist(),
        'loadings': loadings.tolist(),
        'cluster_labels': cluster_labels.tolist()
    }
    return result
