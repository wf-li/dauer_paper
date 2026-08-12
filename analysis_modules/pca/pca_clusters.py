import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from .pca import run_pca
from ..src.data_manager import DataManager
from ..src.utils import exclude_postemb_edgetable
from pathlib import Path

def get_kmeans_labels(adjs, n_clusters=2, random_state=47):
    similarity_matrix = cosine_similarity(adjs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    return cluster_labels

if __name__ == "__main__":
    datatype = 'connectome'
    output_path = Path('analysis_modules/pca/outputs/')
    output_file = Path(f'pca_{datatype}.json')
    pca_components = 2
    random_seed = 47
    kmeans_clusters = 2

    data = DataManager(
        data_path='data',
        include_postemb=True,
        include_muscle=True,
        npair_result=True
    )

    edgetable = data.get_data_edgetable(datatype=datatype)

    # include datasets
    datasets = [
        'L1-1','L1-2','L1-3','L1-4',
        'L2','L3','adult-1','adult-2',
        'dauer-1','dauer-2','dauer-daf2'
    ]
    edgetable = edgetable.loc[:,datasets]
    edgetable = edgetable.replace(np.nan,0)

    # exclude postembryonic neurons from index
    edgetable = exclude_postemb_edgetable(edgetable, datasets)

    features = np.array([edgetable[col].values for col in edgetable.columns])
    output_data = run_pca(edgetable)
    cluster_labels = get_kmeans_labels(features, kmeans_clusters, random_seed)
    output_data['cluster_labels'] = cluster_labels.tolist()

    output_path.mkdir(parents = True, exist_ok=True)
    with open(output_path / output_file, 'w+') as fp:
        json.dump(output_data, fp, indent=4)
