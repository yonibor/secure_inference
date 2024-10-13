import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch


from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning

from research.clustering.crelu_block import ClusterRelu, create_default_prototype

class ClusterConvergenceException(Exception):
    def __init__(self, message='convergence exception') -> None:
        super().__init__(message)


def _format_cluster_samples(drelu_maps):
    if isinstance(drelu_maps, torch.Tensor):
        drelu_maps = drelu_maps.numpy()
    assert isinstance(drelu_maps, np.ndarray) and drelu_maps.ndim == 3, 'incorrect input format'
    samples = drelu_maps.reshape(drelu_maps.shape[0], -1).T
    return samples


def cluster_neurons(drelu_maps, no_converge_fail=True, precompute_affinity=True,
                    preference_quantile=None):
    results = {'all_zero': False, 'cluster_res': None, 
               'failed_to_converge':False, 'same_label_affinity': None,
               'diff_label_affinity': None}
    if not torch.any(drelu_maps):
        results['all_zero'] = True
        results['same_label_affinity'] = results['diff_label_affinity'] = 0
        return results
    assert not (not precompute_affinity and preference_quantile is not None), \
        'to get the prefrence you need to precomutpe the affinity matrix'
    samples = _format_cluster_samples(drelu_maps)
    affinity_mat = -pairwise_distances(samples, metric='hamming')
    preference = None
    if precompute_affinity:
        algo_input = affinity_mat
        input_type = 'precomputed'
        if preference_quantile is not None:
            preference = np.quantile(affinity_mat, preference_quantile)
    else:
        algo_input = samples
        input_type = 'euclidean'
    with warnings.catch_warnings(record=True) as caught_warnings:
        results['cluster_res'] = AffinityPropagation(random_state=42, affinity=input_type,
                                          preference=preference).fit(algo_input)
        for warning in caught_warnings:
            if warning.category == ConvergenceWarning and no_converge_fail:
                results['failed_to_converge'] = True
        if not results['failed_to_converge']:
            results['same_label_affinity'], results['diff_label_affinity'] = \
                _add_mean_dist(results['cluster_res'], affinity_mat)
    return results


def _add_mean_dist(cluster_res, affinity_mat):
    labels = cluster_res.labels_
    same_label_mask = labels[:, None] == labels[None, :]
    same_label_affinity = affinity_mat[same_label_mask].mean()
    diff_label_affinity = affinity_mat[~same_label_mask].mean()
    return same_label_affinity, diff_label_affinity
    

def format_clusters(C, H, W, channel_clusters={}, all_clusters=True) -> ClusterRelu:
    assert not (all_clusters and any(c not in channel_clusters for c in range(C))), \
        'need to contain all channels if all clusters is passed'
    prototype = create_default_prototype(C=C, H=H, W=W)
    active_channels = []
    for channel, cluster_details in channel_clusters.items():
        if not cluster_details['all_zero']:
            active_channels.append(channel)
        if cluster_details['all_zero'] or cluster_details['failed_to_converge']:
            continue
        cluster_res = cluster_details['cluster_res']
        cluster_centers_indices = cluster_res.cluster_centers_indices_
        labels = cluster_res.labels_.reshape(H, W)
        for label, cluster_idx in enumerate(cluster_centers_indices):
            label_rows, label_cols = np.nonzero(labels == label)
            center_row = cluster_idx // W 
            center_col = cluster_idx % W
            prototype[0, channel, label_rows, label_cols] = center_row
            prototype[1, channel, label_rows, label_cols] = center_col
    active_channels = torch.Tensor(sorted(active_channels))
    return prototype, active_channels


def plot_drelu(drelu, save_path=None, title=None):
    plt.close("all")
    plt.figure(1)
    plt.clf()
    plt.imshow(np.squeeze(drelu), origin='lower')
    plt.colorbar()
    
    if title is not None:
        plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



def plot_clustering(cluster_res, H, W, save_path=None,
                    title=None):
    if cluster_res is None:
        return
    cluster_centers_indices = cluster_res.cluster_centers_indices_
    labels = cluster_res.labels_

    n_clusters_ = len(cluster_centers_indices)
    
    X = np.meshgrid(np.arange(H), np.arange(W))
    X = np.stack([X[0].ravel(), X[1].ravel()], axis=1)

    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, n_clusters_)))

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.scatter(
            X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
        )
        plt.scatter(
            cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
        )
        for x in X[class_members]:
            plt.plot(
                [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
            )
    prefix = f"Estimated number of clusters: {n_clusters_}" 
    if title is not None:
        title = f'{prefix}\n{title}'
    else:
        title = prefix
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)