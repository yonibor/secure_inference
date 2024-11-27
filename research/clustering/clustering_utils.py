import warnings
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from research.clustering.model.crelu_block import (
    create_default_labels,
    create_default_prototype,
)


class ClusterConvergenceException(Exception):
    def __init__(self, message="convergence exception") -> None:
        super().__init__(message)


def format_cluster_samples(drelu_maps: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(drelu_maps, torch.Tensor):
        drelu_maps = drelu_maps.numpy()
    assert (
        isinstance(drelu_maps, np.ndarray) and drelu_maps.ndim == 4
    ), "incorrect input format"
    samples = drelu_maps.reshape(drelu_maps.shape[0], -1).T
    return samples


def get_affinity_mat(drelu_maps: np.ndarray) -> np.ndarray:
    samples = format_cluster_samples(drelu_maps)
    affinity_mat = -pairwise_distances(samples, metric="hamming")
    return affinity_mat


def get_default_cluster_details(**kwargs) -> dict:
    details = dict(
        clusters=None,
        channels=None,
        id=id,
        all_zero=False,
        failed_to_converge=False,
        same_label_affinity=0,
        diff_label_affinity=0,
        prototype_affinity=0,
    )
    details.update(kwargs)
    return details


def _verify_keys(details):
    assert set(details.keys()) == set(get_default_cluster_details().keys())


def cluster_neurons(
    drelu_maps: np.ndarray,
    prev_cluster_details: dict,
    no_converge_fail: bool = True,
    precompute_affinity: bool = True,
    preference_quantile: Optional[float] = None,
) -> Dict:
    results = get_default_cluster_details(
        clusters=prev_cluster_details["clusters"],
        id=prev_cluster_details["id"],
        channels=prev_cluster_details["channels"],
    )
    if not np.any(drelu_maps):
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
    (
        results["same_label_affinity"],
        results["diff_label_affinity"],
        results["prototype_affinity"],
    ) = get_mean_dist(results.get("clusters"), affinity_mat)
    _verify_keys(results)
    return results


def get_mean_dist(clusters: AffinityPropagation, affinity_mat: np.ndarray):
    same_label_affinity = diff_label_affinity = prototype_affinity = None
    if clusters is None:
        return same_label_affinity, diff_label_affinity, prototype_affinity
    labels = clusters.labels_
    same_label_mask = labels[:, None] == labels[None, :]
    if same_label_mask.any():
        same_label_affinity = affinity_mat[same_label_mask].mean()
    if not same_label_mask.all():
        diff_label_affinity = affinity_mat[~same_label_mask].mean()
    prototype_indices = clusters.cluster_centers_indices_[labels]
    prototype_affinity = affinity_mat[np.arange(len(labels)), prototype_indices].mean()
    return same_label_affinity, diff_label_affinity, prototype_affinity


def format_clusters(
    C: int,
    H: int,
    W: int,
    clusters_details: Dict[str, Dict] = {},
    all_clusters: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_channels = np.concatenate([details["channels"] for details in clusters_details])
    assert not (
        all_clusters and set(all_channels) != set(range(C))
    ), "need to contain all channels if all clusters is passed"
    prototype = create_default_prototype(C=C, H=H, W=W)
    labels = create_default_labels(C=C, H=H, W=W)
    crelu_channels, original_relu_channels = np.full((C,), False), np.full((C,), False)
    for cur_details in clusters_details:
        clusters = cur_details["clusters"]
        channels: np.ndarray = cur_details["channels"]
        if not cur_details["all_zero"]:
            if clusters is None:
                original_relu_channels[channels] = True
            else:
                crelu_channels[channels] = True
        if (
            cur_details["all_zero"]
            or cur_details["failed_to_converge"]
            or (clusters is None)
        ):
            continue
        cluster_centers_indices = clusters.cluster_centers_indices_
        cur_labels = clusters.labels_.reshape(channels.size, H, W)
        labels[channels] = torch.from_numpy(cur_labels)
        for label, cluster_idx in enumerate(cluster_centers_indices):
            label_local_channels, label_rows, label_cols = np.nonzero(
                cur_labels == label
            )
            label_channels = channels[label_local_channels]
            center_local_channel, center_row, center_col = np.unravel_index(
                cluster_idx, cur_labels.shape
            )
            center_channel = channels[center_local_channel]
            prototype[0, label_channels, label_rows, label_cols] = center_channel
            prototype[1, label_channels, label_rows, label_cols] = center_row
            prototype[2, label_channels, label_rows, label_cols] = center_col
    return prototype, crelu_channels, original_relu_channels, labels


def cluster_channels_kmeans(drelu_maps: np.ndarray, k: int):
    examples = drelu_maps.transpose(1, 0, 2, 3)  # shape channel, batch, height, width
    examples = examples.reshape(examples.shape[0], -1).astype(np.float32)
    clusters = KMeans(n_clusters=k).fit(examples)

    channels = [
        np.nonzero(clusters.labels_ == label)[0]
        for label in np.unique(clusters.labels_)
    ]
    return channels


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
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[np.ndarray] = None,
):
    if clusters is None:
        return
    cluster_centers_indices = clusters.cluster_centers_indices_
    if labels is None:
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
