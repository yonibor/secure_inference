from math import ceil
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, plot_tree


def _get_features_single_neuron(
    classifier: DecisionTreeClassifier,
    target: np.ndarray,
    features: np.ndarray,
    features_amount: int,
):
    classifier.fit(features, target)
    top_features = np.argsort(classifier.feature_importances_)[-features_amount:][::-1]
    if top_features.shape[0] < features_amount:
        repeat_amount = int(ceil(features_amount / top_features.shape[0]))
        top_features = np.tile(top_features, repeat_amount)[:features_amount]
    return top_features, classifier.feature_importances_[top_features]


def get_all_features(
    samples: np.ndarray, centers_indices: np.ndarray, depth: int, features_amount: int
):
    classifier = DecisionTreeClassifier(random_state=42, max_depth=depth)
    chosen_features, features_importance = [], []
    features = samples[:, centers_indices]
    # results = Parallel(n_jobs=-1)(
    #     delayed(_get_features_single_neuron)(
    #         samples[:, neuron_idx], features, depth, features_amount
    #     )
    #     for neuron_idx in range(samples.shape[1])
    # )
    for neuron_idx in range(samples.shape[1]):
        cur_features_indices, cur_feature_importances = _get_features_single_neuron(
            classifier=classifier,
            target=samples[:, neuron_idx],
            features=features,
            features_amount=features_amount,
        )
        # cur_features_indices, cur_feature_importances = results[neuron_idx]
        cur_features = centers_indices[cur_features_indices]
        chosen_features.append(cur_features)
        features_importance.append(cur_feature_importances)
    chosen_features = np.array(chosen_features)
    features_importance = np.array(features_importance)
    return chosen_features, features_importance


def get_features_counts(samples: np.ndarray, features: np.ndarray) -> np.ndarray:
    features_amount = features.shape[-1]
    samples.dtype == np.int
    samples_features = samples[:, features]
    truth_table = np.unpackbits(
        np.arange(2**features_amount, dtype=np.uint8)[:, None], axis=1
    )
    truth_table = truth_table[:, -features_amount:]
    truth_table_masks = np.all(
        samples_features[..., None, :] == truth_table[None, None, ...], axis=-1
    ).astype(int)
    active_count = np.einsum("ab,abc->bc", samples, truth_table_masks)
    not_active_count = np.einsum("ab,abc->bc", 1 - samples, truth_table_masks)
    counts = np.stack([not_active_count, active_count], axis=-1)
    # decisions = counts.argmax(axis=-1)
    return counts


def create_decisions_map(counts: np.ndarray, method: str) -> np.ndarray:
    assert method in ["majority", "copy", "ratio"]
    if method == "majority":
        decisions = counts.argmax(axis=-1)
    elif method == "ratio":
        total = counts.sum(axis=-1)
        decisions = counts[..., 1] / (total + 1e-9)
    elif method == "copy":
        assert (
            counts.shape[-2] == 2
        ), "only works for single prototype, that have two options- active or not active"
        decisions = np.tile([0, 1], (counts.shape[0], 1))
    decisions = decisions.astype(np.float32)
    return decisions


def calc_accuracy(counts: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    decisions_rounded = decisions.round()

    id_counts = counts[..., 1]
    id_succ = np.sum(id_counts * (decisions_rounded == 1))
    id_counts_sum = id_counts.sum()
    id_accuracy = id_succ / id_counts_sum

    zero_counts = counts[..., 0]
    zero_succ = np.sum(zero_counts * (decisions_rounded == 0))
    zero_counts_sum = zero_counts.sum()
    zero_accuracy = zero_succ / zero_counts_sum

    total_accuracy = (zero_succ + id_succ) / (id_counts_sum + zero_counts_sum)

    return total_accuracy, id_accuracy, zero_accuracy
