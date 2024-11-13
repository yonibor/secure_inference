import copy
import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from research.clustering.crelu_block import (
    create_default_labels,
    create_default_prototype,
)


class ClusterConvergenceException(Exception):
    def __init__(self, message="convergence exception") -> None:
        super().__init__(message)


def format_cluster_samples(drelu_maps: np.ndarray) -> np.ndarray:
    if isinstance(drelu_maps, torch.Tensor):
        drelu_maps = drelu_maps.numpy()
    assert (
        isinstance(drelu_maps, np.ndarray) and drelu_maps.ndim == 3
    ), "incorrect input format"
    samples = drelu_maps.reshape(drelu_maps.shape[0], -1).T
    return samples


def get_affinity_mat(drelu_maps: np.ndarray) -> np.ndarray:
    samples = format_cluster_samples(drelu_maps)
    affinity_mat = -pairwise_distances(samples, metric="hamming")
    return affinity_mat


def get_default_cluster_details(
    clusters: Optional[AffinityPropagation] = None,
) -> dict:
    details = dict(
        clusters=clusters,
        all_zero=False,
        failed_to_converge=False,
        same_label_affinity=0,
        diff_label_affinity=0,
    )
    return details


def _verify_keys(details):
    assert set(details.keys()) == set(get_default_cluster_details().keys())


def cluster_neurons(
    drelu_maps: np.ndarray,
    prev_clusters: AffinityPropagation,
    no_converge_fail: bool = True,
    precompute_affinity: bool = True,
    preference_quantile: Optional[float] = None,
) -> Dict:
    results = get_default_cluster_details(clusters=prev_clusters)
    if not torch.any(drelu_maps):
        results["all_zero"] = True
        _verify_keys(results)
        return results

    assert not (
        not precompute_affinity and preference_quantile is not None
    ), "to get the prefrence you need to precomutpe the affinity matrix"
    affinity_mat = get_affinity_mat(drelu_maps)
    preference = None
    if precompute_affinity:
        algo_input = affinity_mat
        input_type = "precomputed"
        if preference_quantile is not None:
            preference = np.quantile(affinity_mat, preference_quantile)
    else:
        algo_input = format_cluster_samples(drelu_maps)
        input_type = "euclidean"
    with warnings.catch_warnings(record=True) as caught_warnings:
        clusters = AffinityPropagation(
            random_state=42,
            affinity=input_type,
            preference=preference,
            max_iter=300,
        ).fit(algo_input)
        for warning in caught_warnings:
            if warning.category == ConvergenceWarning and no_converge_fail:
                results["failed_to_converge"] = True

    if not results["failed_to_converge"]:
        results["clusters"] = clusters
    results["same_label_affinity"], results["diff_label_affinity"] = get_mean_dist(
        results.get("clusters"), affinity_mat
    )
    _verify_keys(results)
    return results


def get_mean_dist(clusters: AffinityPropagation, affinity_mat: np.ndarray):
    same_label_affinity, diff_label_affinity = None, None
    if clusters is None:
        return same_label_affinity, diff_label_affinity
    labels = clusters.labels_
    same_label_mask = labels[:, None] == labels[None, :]
    if same_label_mask.any():
        same_label_affinity = affinity_mat[same_label_mask].mean()
    if not same_label_mask.all():
        diff_label_affinity = affinity_mat[~same_label_mask].mean()
    return same_label_affinity, diff_label_affinity


def format_clusters(
    C: int,
    H: int,
    W: int,
    channel_clusters: Dict[str, Dict] = {},
    all_clusters: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not (
        all_clusters and any(c not in channel_clusters for c in range(C))
    ), "need to contain all channels if all clusters is passed"
    prototype = create_default_prototype(C=C, H=H, W=W)
    labels = create_default_labels(C=C, H=H, W=W)
    crelu_channels, original_relu_channels = [], []
    for channel, cluster_details in channel_clusters.items():
        clusters = cluster_details["clusters"]
        if not cluster_details["all_zero"]:
            if clusters is None:
                original_relu_channels.append(channel)
            else:
                crelu_channels.append(channel)
        if (
            cluster_details["all_zero"]
            or cluster_details["failed_to_converge"]
            or (clusters is None)
        ):
            continue
        cluster_centers_indices = clusters.cluster_centers_indices_
        channel_labels = clusters.labels_.reshape(H, W)
        labels[channel] = torch.from_numpy(channel_labels)
        for label, cluster_idx in enumerate(cluster_centers_indices):
            label_rows, label_cols = np.nonzero(channel_labels == label)
            center_row = cluster_idx // W
            center_col = cluster_idx % W
            prototype[0, channel, label_rows, label_cols] = center_row
            prototype[1, channel, label_rows, label_cols] = center_col
    crelu_channels = sorted(crelu_channels)
    original_relu_channels = sorted(original_relu_channels)
    return prototype, crelu_channels, original_relu_channels, labels


def plot_drelu(
    drelu: np.ndarray, save_path: Optional[str] = None, title: Optional[str] = None
):
    plt.close("all")
    plt.figure(1)
    plt.clf()
    plt.imshow(np.squeeze(drelu), origin="lower")
    plt.colorbar()

    if title is not None:
        plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_clustering(
    clusters: AffinityPropagation,
    H: int,
    W: int,
    save_path=Optional[str],
    title=Optional[str],
):
    if clusters is None:
        return
    cluster_centers_indices = clusters.cluster_centers_indices_
    labels = clusters.labels_

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
        title = f"{prefix}\n{title}"
    else:
        title = prefix
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
