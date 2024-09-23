import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning

class ClusterConvergenceException(Exception):
    def __init__(self, message='convergence exception') -> None:
        super().__init__(message)


def _format_cluster_samples(drelu_maps):
    if isinstance(drelu_maps, torch.Tensor):
        drelu_maps = drelu_maps.numpy()
    if drelu_maps.ndim == 5:
        drelu_maps = drelu_maps.reshape(-1, *drelu_maps.shape[2:])
    assert isinstance(drelu_maps, np.ndarray) and drelu_maps.ndim == 4, 'incorrect input format'
    samples = drelu_maps.reshape(drelu_maps.shape[0], -1).T  # TODO: change when moving to multi channels
    return samples


def cluster_neurons(drelu_maps, no_converge_fail=True, precompute_affinity=True,
                    preference_quantile=None):
    assert not (not precompute_affinity and preference_quantile is not None), \
        'to get the prefrence you need to precomutpe the affinity matrix'
    samples = _format_cluster_samples(drelu_maps)
    preference = None
    if precompute_affinity:
        affinity_mat = -pairwise_distances(samples, metric='hamming')
        algo_input = affinity_mat
        input_type = 'precomputed'
        if preference_quantile is not None:
            preference = np.quantile(affinity_mat, preference_quantile)
    else:
        algo_input = samples
        input_type = 'euclidean'
    with warnings.catch_warnings(record=True) as caught_warnings:
        cluster_res = AffinityPropagation(random_state=42, affinity=input_type,
                                          preference=preference).fit(algo_input)
        for warning in caught_warnings:
            if warning.category == ConvergenceWarning and no_converge_fail:
                raise ClusterConvergenceException(warning.message)
    return cluster_res


def plot_clustering(cluster_res, H, W):
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

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()