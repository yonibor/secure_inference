import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from research.clustering.clustering_utils import (
    cluster_channels_kmeans,
    cluster_neurons,
    find_features,
    format_cluster_samples,
    format_clusters,
    get_affinity_mat,
    get_decisions_data,
    get_default_clusters_details,
    get_mean_dist,
)
from research.clustering.model.crelu_block import ClusterRelu
from research.clustering.model.model_handling import get_layer
from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import (
    Params as resnet18_8xb16_cifar100_Params,
)


class CreluManager:
    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        config: dict,
        preference: Dict[int, float],
        id_channels: List[int],
        keep_relu_channels: List[int],
        C: int,
        H: int,
        W: int,
        batch_size: int,
        cluster_no_converge_fail: bool = True,
        use_cluster_mean: bool = False,
        use_crelu_existing_params: bool = False,
    ) -> None:
        self.backbone = model.backbone
        self.layer_name = layer_name
        self.crelu = get_layer(self.backbone, self.layer_name)
        self.config = self._format_config(config)
        self.C, self.H, self.W = C, H, W
        self.batch_size = batch_size
        self.cluster_no_converge_fail = cluster_no_converge_fail
        self.use_cluster_mean = use_cluster_mean
        self.use_crelu_existing_params = use_crelu_existing_params
        self.preference = preference
        self.id_channels = id_channels
        # self.id_channels = [
        #     c for c in range(self.C) if c != 0
        # ]  # TODO yoni: remove!!!!!!!!!!!!!!!
        # self.id_channels = list(range(self.C))  # TODO yoni: remove!!!!!!!!!!!!!!!
        self.keep_relu_channels = keep_relu_channels

        self.batch_idx = 0
        self.drelu_maps = self.prev_drelu_maps = None
        self.drelu_maps_counter = 0
        self.cur_clusters_details = [
            get_default_clusters_details(channels=np.array([channel]), id=channel)
            for channel in range(self.C)
        ]
        self.cur_mean_drelu_maps = None

        self._init_crelu()

    def hook_fn(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        if not module.training and self.config["general"]["only_during_training"]:
            return

        self.batch_idx += 1

        self._update_id_channels()
        self._run_clustering_flow(input)
        self._update_sigmoid()

    def _run_clustering_flow(self, activation_input) -> None:
        if not self.config["cluster"]["use"]:
            return
        if self.batch_idx == self.config["cluster"]["start"]:
            print(
                f"---------------- {self.layer_name}: start clustering ----------------"
            )
        if self.should_update_drelu_stats():
            self._update_drelu_maps(activation_input)
        else:
            self._reset_drelu_maps()

        if self.should_update_clusters():
            self._update_clusters()

        self._update_inter()

        if self.should_update_post_stats():
            self._update_post_stats()

    def _update_id_channels(self) -> None:
        config = self.config["id_warmup"]
        if config["use"] and self.batch_idx == config["start"]:
            print(
                f"---------------- {self.layer_name}, batch {self.batch_idx}: start id layers ----------------"
            )
            self.crelu.original_relu_channels[self.id_channels] = False
            self.crelu.id_channels[self.id_channels] = True

    def _update_sigmoid(self) -> None:
        self._update_to_sigmoid()
        self._update_from_sigmoid()

    def _update_to_sigmoid(self) -> None:
        config = self.config["sigmoid"]["to_sigmoid"]
        if not self.config["sigmoid"]["use"] or self.batch_idx < config["start"]:
            return
        self.crelu.use_sigmoid = True
        if self.batch_idx > config["end"]:
            return

        sigmoid_factor = self._interpulate_scheduler(
            start_step=config["start"],
            end_step=config["end"],
            start_value=self.config["sigmoid"]["relu_factor"],
            end_value=self.config["sigmoid"]["cluster_factor"],
        )
        self.crelu.sigmoid_factor = sigmoid_factor

    def _update_from_sigmoid(self) -> None:
        if not self.config["sigmoid"]["use"]:
            return
        config = self.config["sigmoid"]["from_sigmoid"]
        if self.batch_idx < config["start"]:
            return
        if self.batch_idx >= config["end"]:
            self.crelu.use_sigmoid = False
        else:
            sigmoid_factor = self._interpulate_scheduler(
                start_step=config["start"],
                end_step=config["end"],
                start_value=self.config["sigmoid"]["cluster_factor"],
                end_value=self.config["sigmoid"]["relu_factor"],
            )
            self.crelu.sigmoid_factor = sigmoid_factor

    @staticmethod
    def _format_config(config: dict) -> dict:
        config = deepcopy(config)
        cluster = config["cluster"]
        cluster["wait"] = config["drelu_stats"]["batch_amount"]

        to_sigmoid = config["sigmoid"]["to_sigmoid"]
        from_sigmoid = config["sigmoid"]["from_sigmoid"]
        to_sigmoid["use"] = from_sigmoid["use"] = config["sigmoid"]["use"]

        tasks = dict(
            warmup=config["warmup"],
            id_warmup=config["id_warmup"],
            to_sigmoid=to_sigmoid,
            cluster=cluster,
            from_sigmoid=from_sigmoid,
        )
        prev_end = 0
        for name, task in tasks.items():
            use_task = task.get("use", True)
            if use_task:
                task["start"] = prev_end + task.get("wait", 0)
                task["end"] = task["start"] + task["iters"]
                prev_end = task["end"] + task.get("cooldown", 0)
        config["general"]["end"] = prev_end
        return config

    @staticmethod
    def get_iters_end(config: dict) -> int:
        config = CreluManager._format_config(config)
        return config["general"]["end"], config["cluster"].get(
            "end", config["warmup"]["end"]
        )

    def should_update_drelu_stats(self) -> bool:
        batch_amount = self.config["drelu_stats"]["batch_amount"]
        for next_batch in range(self.batch_idx, self.batch_idx + batch_amount):

            if self._should_update(
                start=self.config["cluster"].get("start"),
                end=None,
                update_freq=self.config["cluster"]["update_freq"],
                use=self.config["cluster"]["use"],
                batch_idx=next_batch,
            ):
                return True
        return False

    def should_update_clusters(self) -> bool:
        config = self.config["cluster"]
        return self._should_update(
            start=config.get("start"),
            end=config.get("end"),
            use=config["use"],
            update_freq=config["update_freq"],
        )

    def should_update_post_stats(self) -> bool:
        config = self.config["cluster"]
        if not config["use"]:
            return False
        after_clustering = self.batch_idx >= config["end"]
        return after_clustering and self._should_update(
            start=config.get("start"),
            end=None,
            use=config["use"],
            update_freq=config["update_freq"],
        )

    def _should_update(
        self,
        start: Optional[int],
        end: Optional[int],
        update_freq: int,
        use: bool,
        batch_idx: Optional[int] = None,
    ) -> bool:
        if not use:
            return False
        if batch_idx is None:
            batch_idx = self.batch_idx
        batch_after_start = batch_idx - start
        if (
            update_freq is None
            or batch_after_start < 0
            or (end is not None and batch_idx >= end)
        ):
            return False
        return batch_after_start % update_freq == 0

    def _update_drelu_maps(self, activation_input: torch.Tensor) -> None:
        cur_drelu_map = activation_input[0].gt(0).cpu().detach().numpy()
        if self.drelu_maps is None:
            self.drelu_maps = cur_drelu_map
        else:
            self.drelu_maps = np.concatenate([self.drelu_maps, cur_drelu_map], axis=0)
        self.cur_mean_drelu_maps = np.mean(self.drelu_maps, axis=0)
        self.drelu_maps_counter += 1
        assert self.drelu_maps_counter <= self.config["drelu_stats"]["batch_amount"]

    def _should_cluster_channel(self, clusters_details: dict) -> bool:
        assert (
            not self.config["group_channels"]["group"]
            and len(clusters_details["channels"]) == 1
        ), "currently not supported with filtering channles"

        channel = clusters_details["channels"][0]
        no_cluster_channels = self.id_channels + self.keep_relu_channels
        should_cluster = channel not in no_cluster_channels
        return should_cluster

    def _update_single_group_clusters(self, clusters_details: dict) -> None:
        samples = format_cluster_samples(
            self.drelu_maps[:, clusters_details["channels"]]
        )
        finished_clustering = (
            clusters_details["clusters"] is not None
            and self.config["cluster"]["cluster_once"]
        )
        should_cluster = self._should_cluster_channel(clusters_details)

        if not finished_clustering and should_cluster:
            affinity_mat = get_affinity_mat(samples)
            preference = self.preference[clusters_details["channels"][0]]
            labels, centers_indices, all_zero, failed_to_converge = cluster_neurons(
                samples=samples,
                preference_quantile=preference,
                no_converge_fail=self.cluster_no_converge_fail,
                affinity_mat=affinity_mat,
            )
            clusters_details.update(
                all_zero=all_zero, failed_to_converge=failed_to_converge
            )
            if not failed_to_converge:
                clusters_details["clusters"] = dict(
                    labels=labels,
                    centers_indices=centers_indices,
                )
                clusters_details["features_data"] = find_features(
                    samples=samples,
                    labels=labels,
                    centers_indices=centers_indices,
                    find_best_features=self.config["best_features"]["use"],
                    features_depth=self.config["best_features"]["depth"],
                    features_amount=self.config["best_features"]["amount"],
                    tree_positive_weight=self.config["best_features"][
                        "tree_positive_weight"
                    ],
                )

                mean_dist_details = get_mean_dist(
                    labels=labels,
                    centers_indices=centers_indices,
                    affinity_mat=affinity_mat,
                )
                clusters_details.update(mean_dist_details)
            else:
                print(
                    f"Caught convergence warning at batch {self.batch_idx} ",
                    f"layer {self.layer_name}, id {clusters_details['id']}\n",
                    "not updating clusters",
                )

        if should_cluster and clusters_details["features_data"] is not None:
            clusters_details["features_data"].update(
                get_decisions_data(
                    samples=samples,
                    features=clusters_details["features_data"]["features"],
                    find_best_features=self.config["best_features"]["use"],
                    decision_method=self.config["best_features"]["method"],
                    **self.config["best_features"]["method_args"],
                )
            )

    def _reset_drelu_maps(self) -> None:
        if self.drelu_maps is not None:
            self.prev_drelu_maps = self.drelu_maps
            self.drelu_maps = None
            self.drelu_maps_counter = 0

    def _create_grouped_clusters_templates(self):
        config = self.config["group_channels"]
        if not config["group"] or (
            self.config["cluster"]["cluster_once"] and self.cluster_started
        ):
            return
        print("creating group templates")
        groups_channels = []
        self.cur_clusters_details = []
        zero_channels_mask = np.logical_not(self.drelu_maps).all(axis=(0, 2, 3))
        zero_channels = np.nonzero(zero_channels_mask)[0]
        groups_channels.append(zero_channels)
        remaining_channels = np.nonzero(np.logical_not(zero_channels_mask))[0]
        if not config["group_by_kmeans"]:
            groups_channels.append(remaining_channels)
        else:
            remaining_drelu = self.drelu_maps[:, remaining_channels]
            kmeans_local_channels = cluster_channels_kmeans(
                remaining_drelu, config["k"]
            )
            kmeans_channels = [
                remaining_channels[channels] for channels in kmeans_local_channels
            ]
            groups_channels.extend(kmeans_channels)
        for channels in groups_channels:
            if channels.size > 0:
                details = get_default_clusters_details(
                    channels=channels, id=len(self.cur_clusters_details)
                )
                self.cur_clusters_details.append(details)

    def _update_clusters(self) -> None:
        assert (
            self.drelu_maps.shape[0]
            <= self.config["drelu_stats"]["batch_amount"] * self.batch_size
        )
        self._create_grouped_clusters_templates()

        from datetime import datetime

        start_time = datetime.now()
        print(f"start clustering {self.layer_name}, batch {self.batch_idx}")
        for clusters_details in self.cur_clusters_details:
            self._update_single_group_clusters(clusters_details)
        print(f"end clustering {datetime.now() - start_time}")

        features_amount = (
            1
            if not self.config["best_features"]["use"]
            else self.config["best_features"]["amount"]
        )
        prototype, decisions, crelu_channels, original_relu_channels, labels = (
            format_clusters(
                C=self.C,
                H=self.H,
                W=self.W,
                features_amount=features_amount,
                clusters_details=self.cur_clusters_details,
                id_channels=self.id_channels,
            )
        )
        original_relu_channels[self.keep_relu_channels] = True
        self.crelu.prototype = prototype
        self.crelu.decisions = decisions
        self.crelu.labels = labels
        self.crelu.crelu_channels = crelu_channels
        self.crelu.original_relu_channels = original_relu_channels

        # print(f"saving path for {self.layer_name}")
        # dir = "/workspaces/secure_inference/tests/26_11_multi_prototype/all_stats/drelu_per_layer/"
        # os.makedirs(dir, exist_ok=True)
        # out_path = os.path.join(
        #     dir,
        #     f"{self.layer_name}.npy",
        # )
        # np.save(out_path, self.drelu_maps)

        self._reset_drelu_maps()

    def _interpulate_scheduler(self, start_step, end_step, start_value, end_value):
        if self.batch_idx <= start_step:
            return start_value
        elif self.batch_idx >= end_step:
            return end_value
        elif start_step == end_step == self.batch_idx:
            assert start_value == end_value
            return start_value
        relative_position = (self.batch_idx - start_step) / (end_step - start_step)
        inter_value = start_value + relative_position * (end_value - start_value)
        return inter_value

    def _update_inter(self) -> None:
        new_inetr = self._interpulate_scheduler(
            start_step=self.config["cluster"]["start"],
            end_step=self.config["cluster"]["end"],
            start_value=self.config["inter"]["start_value"],
            end_value=self.config["inter"]["end_value"],
        )
        self.crelu.inter = new_inetr

    def _init_crelu(self) -> None:
        self.crelu.use_cluster_mean = self.use_cluster_mean
        self.crelu.is_dummy = False
        self.crelu.inter_before_activation = self.config["inter"]["before_activation"]
        self.crelu.multi_prototype = self.config["best_features"]["use"]

        if not self.use_crelu_existing_params:
            self.crelu.set_default_values(set_size=False)
            inter = self.config["inter"]["start_value"]
            self.crelu.inter = inter
        else:
            self.crelu.verify_buffers_init()

        self._update_sigmoid()

    def _update_post_stats(self) -> None:
        for clusters_details in self.cur_clusters_details:
            clusters = clusters_details.get("clusters")
            if clusters is None:
                continue
            samples = format_cluster_samples(
                self.drelu_maps[:, clusters_details["channels"]]
            )
            affinity_mat = get_affinity_mat(samples)

            mean_dist_details = get_mean_dist(
                labels=clusters["labels"],
                centers_indices=clusters["centers_indices"],
                affinity_mat=affinity_mat,
            )
            clusters_details.update(mean_dist_details)
        self._reset_drelu_maps()

    @property
    def cluster_started(self):
        return self.batch_idx > self.config["cluster"]["start"]


def add_crelu_hooks(
    model: nn.Module,
    layer_names: List[str],
    layers_args: Optional[Dict[str, dict]] = None,
    **kwargs,
) -> Dict[str, CreluManager]:
    hooks = {}
    layers_args = {} if layers_args is None else layers_args
    for layer_name in layer_names:
        cur_layer_args = layers_args.get(layer_name, {})
        C, H, W = resnet18_8xb16_cifar100_Params().LAYER_NAME_TO_DIMS[layer_name]
        hook_instance = CreluManager(
            model, layer_name, C=C, H=H, W=W, **cur_layer_args, **kwargs
        )
        layer = get_layer(model.backbone, layer_name)
        layer.register_forward_hook(hook_instance.hook_fn)
        hooks[layer_name] = hook_instance
    return hooks
