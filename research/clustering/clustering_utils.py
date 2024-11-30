import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

from research.clustering.model.crelu_block import (
    create_default_decisions,
    create_default_labels,
    create_default_prototype,
)
from research.clustering.multi_prototype_utils import (
    calc_accuracy,
    create_decisions_map,
    get_all_features,
    get_features_counts,
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
    samples = drelu_maps.reshape(drelu_maps.shape[0], -1)
    return samples


def get_affinity_mat(samples: np.ndarray) -> np.ndarray:
    samples_features_first = samples.T
    affinity_mat = -pairwise_distances(samples_features_first, metric="hamming")
    return affinity_mat


def get_weighted_affinity(samples: np.ndarray, positive_weight: float) -> np.ndarray:
    samples_features_first = samples.T
    pairwise_11 = np.dot(samples_features_first, samples_features_first.T)
    pairwise_10 = np.dot(samples_features_first, 1 - samples_features_first.T)
    pairwise_01 = np.dot(1 - samples_features_first, samples_features_first.T)

    total_features = samples_features_first.shape[1]
    pairwise_00 = total_features - (pairwise_11 + pairwise_01 + pairwise_10)

    numerator = pairwise_10 + pairwise_01
    denominator = (
        pairwise_11 + pairwise_10 + pairwise_01 + pairwise_00 / positive_weight
    )
    distance = numerator / denominator
    affinity = -distance
    return affinity


def get_default_clusters_details(**kwargs) -> dict:
    details = dict(
        clusters=None,
        features_data=None,
        channels=None,
        id=None,
        all_zero=False,
        failed_to_converge=False,
        same_label_affinity=0,
        diff_label_affinity=0,
        prototype_affinity=0,
    )
    details.update(kwargs)
    return details


def cluster_neurons(
    samples: np.ndarray,
    affinity_mat: np.ndarray,
    no_converge_fail: bool = True,
    preference_quantile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    labels = centers_indices = None
    all_zero = failed_to_converge = False
    if not np.any(samples):
        all_zero = True
        return labels, centers_indices, all_zero, failed_to_converge
    preference = None
    if preference_quantile is not None:
        preference = np.quantile(affinity_mat, preference_quantile)
    with warnings.catch_warnings(record=True) as caught_warnings:
        clusters = AffinityPropagation(
            random_state=42,
            affinity="precomputed",
            preference=preference,
            max_iter=300,
        ).fit(affinity_mat)
        labels = clusters.labels_
        centers_indices = clusters.cluster_centers_indices_
        for warning in caught_warnings:
            if warning.category == ConvergenceWarning and no_converge_fail:
                failed_to_converge = True
                break
    return labels, centers_indices, all_zero, failed_to_converge


def get_mean_dist(
    labels: np.ndarray,
    centers_indices: np.ndarray,
    affinity_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    labels = labels.squeeze()
    same_label_mask = labels[:, None] == labels[None, :]
    if same_label_mask.any():
        same_label_affinity = affinity_mat[same_label_mask].mean()
    if not same_label_mask.all():
        diff_label_affinity = affinity_mat[~same_label_mask].mean()
    prototype_indices = centers_indices[labels]
    prototype_affinity = affinity_mat[np.arange(len(labels)), prototype_indices].mean()
    return dict(
        same_label_affinity=same_label_affinity,
        diff_label_affinity=diff_label_affinity,
        prototype_affinity=prototype_affinity,
    )


def format_clusters(
    C: int,
    H: int,
    W: int,
    features_amount: int,
    clusters_details: Dict[str, Dict] = {},
    id_channels: List[int] = [],
    all_clusters: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_channels = np.concatenate([details["channels"] for details in clusters_details])
    assert not (
        all_clusters and set(all_channels) != set(range(C))
    ), "need to contain all channels if all clusters is passed"
    prototype = create_default_prototype(C=C, H=H, W=W, features_amount=features_amount)
    decisions = create_default_decisions(C=C, H=H, W=W, features_amount=features_amount)
    labels = create_default_labels(C=C, H=H, W=W)
    crelu_channels_bool = np.full((C,), False)
    original_relu_channels_bool = crelu_channels_bool.copy()

    for cur_details in clusters_details:
        clusters = cur_details["clusters"]
        features_data = cur_details["features_data"]
        channels: np.ndarray = cur_details["channels"]
        assert channels.shape[0] == 1, "currently only supports single channel"
        channel = channels[0]
        is_id_channel = channel in id_channels
        if not (cur_details["all_zero"] or is_id_channel):
            if clusters is None:
                original_relu_channels_bool[channels] = True
            else:
                crelu_channels_bool[channels] = True
        if is_id_channel or cur_details["all_zero"] or (clusters is None):
            continue
        cur_labels = clusters["labels"].reshape(channels.size, H, W)
        cur_decisions = features_data["decisions"].reshape(channels.size, H, W, -1)
        labels[channels] = torch.from_numpy(cur_labels)
        decisions[channels] = torch.from_numpy(cur_decisions).to(decisions.dtype)

        features_unraveled = np.stack(
            np.unravel_index(features_data["features"], (H, W)), axis=0
        )
        features_unraveled = features_unraveled.reshape(
            features_unraveled.shape[0], H, W, -1
        )
        prototype[1:, channels] = torch.from_numpy(features_unraveled).unsqueeze(1)

    return (
        prototype,
        decisions,
        crelu_channels_bool,
        original_relu_channels_bool,
        labels,
    )


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
    labels: np.ndarray,
    cluster_centers_indices: np.ndarray,
    H: int,
    W: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):

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


def find_features(
    samples: np.ndarray,
    labels: np.ndarray,
    centers_indices: np.ndarray,
    find_best_features: bool,
    features_depth: Optional[int] = None,
    features_amount: Optional[int] = None,
    tree_positive_weight=1,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = samples.astype(int)
    if find_best_features:
        chosen_features, features_importance = get_all_features(
            samples,
            centers_indices,
            features_depth,
            features_amount,
            positive_weight=tree_positive_weight,
        )
    else:
        chosen_features = centers_indices[labels, None]
        features_importance = None
    return dict(features=chosen_features, features_importance=features_importance)


def get_decisions_data(
    samples: np.ndarray,
    features: np.ndarray,
    find_best_features: bool,
    decision_method: str,
    **method_args,
) -> Dict[str, np.ndarray]:
    if not find_best_features:
        decision_method = "copy"
    counts = get_features_counts(samples, features)
    decisions = create_decisions_map(counts, decision_method, **method_args)
    accuracy, id_accuracy, zero_accuracy = calc_accuracy(counts, decisions)

    return dict(
        features_counts=counts,
        decisions=decisions,
        accuracy=accuracy,
        id_accuracy=id_accuracy,
        zero_accuracy=zero_accuracy,
    )
