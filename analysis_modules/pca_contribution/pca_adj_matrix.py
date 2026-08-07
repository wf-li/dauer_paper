import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from ..src.data_manager import DataManager

def get_kmeans_labels(adjs, n_clusters=2, random_state=47):
    similarity_matrix = cosine_similarity(adjs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    return cluster_labels

if __name__ == "__main__":
    datatype = 'proximity'
    output_file = f'analysis_modules/pca_contribution/outputs/pca_{datatype}.json'
    pca_components = 2
    random_seed = 47
    kmeans_clusters = 2

    data = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=False,
        npair_result=True
    )

    edgelist = data.get_data_edgelist(datatype=datatype)
    
    features = np.array([edgelist[col].values for col in edgelist.columns])
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features)

    labels = edgelist.columns
    evr = pca.explained_variance_ratio_
    edge_labels = [f'{pre}-{post}' for pre,post in edgelist.index]
    loadings = pca.components_.T
    cluster_labels = get_kmeans_labels(features, kmeans_clusters, random_seed)

    output_data = {
        'features': features_pca.tolist(),
        'labels': labels.tolist(),
        'explained_variance': evr.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'edge_labels': edge_labels,
        'loadings': loadings.tolist()
    }

    with open(output_file, 'w+') as fp:
        json.dump(output_data, fp, indent=4)
