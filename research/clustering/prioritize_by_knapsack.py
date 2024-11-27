import os
import pickle as pkl
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_clustering_stats(stats_dir: str) -> pd.DataFrame:
    stats = {}
    for sub_dir in os.listdir(stats_dir):
        file_path = os.path.join(stats_dir, sub_dir, "per_channel.csv")
        stats[sub_dir] = pd.read_csv(file_path)
    all_stats = [
        cur_stats.assign(preference=name.split("_")[-1])
        for name, cur_stats in stats.items()
    ]
    all_stats = pd.concat(all_stats)
    max_batch = all_stats["batch_index"].max()
    all_stats = all_stats.query("batch_index == @max_batch").copy()
    all_stats["preference"] = all_stats["preference"].astype(float)
    return all_stats


def _get_knapsack_amounts(
    knapsack_resluts: Dict[str, np.ndarray], dims: Dict[str, list]
) -> Dict[str, np.ndarray]:
    amounts = {}
    for name, blocks in knapsack_resluts.items():
        C, H, W = dims[name]
        full_amount = H * W
        blocks_area = np.prod(blocks, axis=1)
        blocks_amount = np.where(blocks_area == 0, 0, full_amount / blocks_area)
        amounts[name] = blocks_amount
    return amounts


def _get_preference(
    knapsack_amount, channel_clustering_amounts: pd.DataFrame, H: int, W: int
):
    if knapsack_amount == 0:
        preference = "id"
        cur_amount = knapsack_amount
    elif knapsack_amount == H * W:
        preference = "relu"
        cur_amount = knapsack_amount
    else:
        channel_clustering_amounts = channel_clustering_amounts.reset_index()
        clustering_dist = (
            channel_clustering_amounts["cluster_amount"] - knapsack_amount
        ).abs()

        closest_idx = (
            clustering_dist - channel_clustering_amounts["preference"]
        ).idxmin()
        closest_row = channel_clustering_amounts.loc[closest_idx]
        preference = closest_row["preference"]
        cur_amount = closest_row["cluster_amount"]
    return preference, cur_amount


def format_per_layer(priority: pd.DataFrame, layers_names: list) -> dict:
    formatted_priority = {}
    for layer_name in layers_names:
        pref_dict = {}
        id_channels = []
        keep_relu_channels = []
        for _, row in priority.query("layer_name == @layer_name").iterrows():
            channel = row["channel"]
            if row["preference"] == "id":
                id_channels.append(channel)
            elif row["preference"] == "relu":
                keep_relu_channels.append(channel)
            else:
                pref_dict[channel] = row["preference"]
        formatted_priority[layer_name] = {
            "preference": pref_dict,
            "id_channels": id_channels,
            "keep_relu_channels": keep_relu_channels,
        }
    return formatted_priority


def prioritize_channels(
    stats_dir: str, knapsack_path: str, dims: Dict[str, list]
) -> pd.DataFrame:
    with open(knapsack_path, "rb") as f:
        knapsack_results = pkl.load(f)
    knapsack_amounts = _get_knapsack_amounts(knapsack_results, dims)
    clustering_amounts = _load_clustering_stats(stats_dir=stats_dir)

    matches = []
    for layer_name, layer_knapsack_amounts in knapsack_amounts.items():
        C, H, W = dims[layer_name]
        for channel in range(C):
            cur_knapsack_amount = layer_knapsack_amounts[channel]
            channel_clustering_amount = clustering_amounts.query(
                "layer_name == @layer_name and channel == @channel"
            )
            match = {
                "layer_name": layer_name,
                "channel": channel,
                "knapsack_amount": cur_knapsack_amount,
            }
            match["preference"], match["cur_amount"] = _get_preference(
                cur_knapsack_amount, channel_clustering_amount, H, W
            )
            matches.append(match)
    matches = pd.DataFrame(matches)
    return matches
