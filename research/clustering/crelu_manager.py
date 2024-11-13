import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from research.clustering.clustering_utils import (
    cluster_neurons,
    format_clusters,
    get_affinity_mat,
    get_default_cluster_details,
    get_mean_dist,
)
from research.clustering.crelu_block import ClusterRelu
from research.clustering.model_handling import get_layer, set_layer
from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import (
    Params as resnet18_8xb16_cifar100_Params,
)


class CreluManager:
    def __init__(
        self,
        model: nn.Module,
        layer_name: str,
        update_config: dict,
        C: int,
        H: int,
        W: int,
        cluster_no_converge_fail: bool = True,
        keep_channels: List[int] = None,
        use_cluster_mean: bool = False,
    ) -> None:
        self.model = model
        self.layer_name = layer_name
        self.update_config = update_config
        self.C, self.H, self.W = C, H, W
        self.cluster_no_converge_fail = cluster_no_converge_fail
        self.use_cluster_mean = use_cluster_mean

        self.prefrence_quantile = self.update_config["cluster"]["preference"][
            "quantile_start"
        ]

        self.batch_idx = None
        self.drelu_maps = self.prev_drelu_maps = None
        self.drelu_maps_counter = 0
        self.cur_cluster_details = [
            get_default_cluster_details(channels=np.array([channel]), id=channel)
            for channel in range(self.C)
        ]
        self.cur_mean_drelu_maps = None
        self.cur_min_inter = self.cur_max_inter = None
        self._first_inter_batch = None
        self.keep_channels = keep_channels if keep_channels is not None else []

        self._init_crelu()
        self.cluster_started = False

    def hook_fn(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        if not module.training and self.update_config["only_during_training"]:
            return

        if self.batch_idx is None:
            self.batch_idx = 0
        else:
            self.batch_idx += 1

        if self.should_update_drelu_stats():
            self._update_drelu_maps(input)
        else:
            self._reset_drelu_maps()

        if self.should_update_clusters():
            self._update_clusters()

        if self.should_update_inter():
            self._update_inter()

        if self.should_update_post_stats():
            self._update_post_stats()

    def should_update_drelu_stats(self) -> bool:
        batch_amount = self.update_config["drelu_stats"]["batch_amount"]
        for next_batch in range(self.batch_idx, self.batch_idx + batch_amount):
            if self._should_update(
                self.update_config["cluster"], next_batch, ignore_max_idx=True
            ):
                return True
        return False

    def should_update_inter(self) -> bool:
        cluster_blocking_inter = (
            self.update_config["inter"]["await_cluster_start"]
            and not self.cluster_started
        )
        update = (
            self._should_update(self.update_config["inter"])
            and not cluster_blocking_inter
        )
        return update

    def should_update_clusters(self) -> bool:
        return self._should_update(self.update_config["cluster"])

    def should_update_post_stats(self) -> bool:
        max_iters = self.update_config.get("max_iters")
        after_max = (
            self.batch_idx is not None
            and max_iters is not None
            and self.batch_idx >= max_iters
        )
        return after_max and self._should_update(
            self.update_config["cluster"], ignore_max_idx=True
        )

    def _should_update(
        self,
        config: dict,
        batch_idx: Optional[int] = None,
        ignore_max_idx: bool = False,
    ) -> bool:
        if batch_idx is None:
            batch_idx = self.batch_idx
        update_freq = config["update_freq"]
        batch_after_warmup = batch_idx - self.update_config["warmup"]
        max_iters = self.update_config.get("max_iters")
        if (
            update_freq is None
            or batch_after_warmup < 0
            or (max_iters is not None and batch_idx >= max_iters and not ignore_max_idx)
        ):
            return False
        elif batch_after_warmup == 0:
            return config["update_on_start"]
        return batch_after_warmup % config["update_freq"] == 0

    def _update_drelu_maps(self, input: torch.Tensor) -> None:
        cur_drelu_map = input[0].gt(0).cpu().detach()
        if self.drelu_maps is None:
            self.drelu_maps = cur_drelu_map
        else:
            self.drelu_maps = torch.concat([self.drelu_maps, cur_drelu_map], dim=0)
        self.cur_mean_drelu_maps = torch.mean(self.drelu_maps.float(), axis=0).numpy()
        self.drelu_maps_counter += 1
        assert (
            self.drelu_maps_counter <= self.update_config["drelu_stats"]["batch_amount"]
        )

    def _get_single_group_clusters(self, prev_cluster_details: dict) -> None:
        finished_clustering = (
            prev_cluster_details["clusters"] is not None
            and self.update_config["cluster"]["cluster_once"]
        )
        if not finished_clustering:
            prev_cluster_details = cluster_neurons(
                self.drelu_maps[:, prev_cluster_details["channels"]],
                prev_cluster_details=prev_cluster_details,
                preference_quantile=self.prefrence_quantile,
                no_converge_fail=self.cluster_no_converge_fail,
            )

            # self.cur_cluster_details[channel] = cluster_neurons(
            #     self.drelu_maps[:, channel],
            #     prev_clusters=self.cur_cluster_details[channel]["clusters"],
            #     preference_quantile=self.prefrence_quantile,
            #     no_converge_fail=self.cluster_no_converge_fail,
            # )

            if prev_cluster_details["failed_to_converge"]:
                print(
                    f"Caught convergence warning at batch {self.batch_idx} ",
                    f"layer {self.layer_name}, id {prev_cluster_details['id']}\n",
                    "not updating clusters",
                )
        return prev_cluster_details

    def _reset_drelu_maps(self) -> None:
        if self.drelu_maps is not None:
            self.prev_drelu_maps = self.drelu_maps
            self.drelu_maps = None
            self.drelu_maps_counter = 0

    def _update_prefrence_quantile(self) -> None:
        if self.prefrence_quantile is not None:
            pref_config = self.update_config["cluster"]["preference"]
            self.prefrence_quantile = max(
                self.prefrence_quantile * pref_config["quantile_decay"],
                pref_config["quantile_min"],
            )

    def _update_clusters(self) -> None:
        if not self.cluster_started:
            self.cluster_started = True
            # # TODO yoni: remove tmp
            # print(f"saving path for {self.layer_name}")
            # out_path = os.path.join(
            #     "/workspaces/secure_inference/tests/12_11_multi_channel/all_stats",
            #     f"{self.layer_name}.npy",
            # )
            # np.save(out_path, self.drelu_maps)

        from datetime import datetime

        start_time = datetime.now()
        print(f"start clustering {self.layer_name}")

        # with concurrent.futures.ThreadPoolExecutor(
        #     max_workers=os.cpu_count()
        # ) as executor:
        #     # Map function_b to items in array concurrently
        #     channels_details = list(executor.map(self._get_channel_clusters, channels))

        new_clusters = []
        for cluster_details in self.cur_cluster_details:
            new_clusters.append(self._get_single_group_clusters(cluster_details))

        print(f"end clustering {datetime.now() - start_time}")
        self.cur_cluster_details = new_clusters

        prototype, crelu_channels, original_relu_channels, labels = format_clusters(
            self.C, self.H, self.W, self.cur_cluster_details
        )
        crelu: ClusterRelu = get_layer(self.model, self.layer_name)
        crelu.prototype = prototype
        crelu.labels = labels
        crelu.crelu_channels = crelu_channels
        crelu.original_relu_channels = original_relu_channels

        self._update_prefrence_quantile()
        self._reset_drelu_maps()

    def _update_inter(self) -> None:
        if self._first_inter_batch is None:
            self._first_inter_batch = self.batch_idx
        crelu = get_layer(self.model, self.layer_name)
        update_step = self._calc_inter_update_step()
        new_inter = crelu.inter + update_step
        new_inter = torch.clamp(new_inter, 0, 1)
        crelu.inter = new_inter
        self._update_inter_stats(crelu)

    def _init_crelu(self) -> None:
        inter = self.update_config["inter"].get("default_start", 0)
        crelu = ClusterRelu(
            C=self.C,
            H=self.H,
            W=self.W,
            inter=inter,
            use_cluster_mean=self.use_cluster_mean,
        )
        crelu.original_relu_channels = torch.arange(self.C)
        crelu.crelu_channels = None
        set_layer(self.model, self.layer_name, crelu)
        self._update_inter_stats(crelu)

    def _update_inter_stats(self, crelu) -> None:
        self.cur_min_inter = torch.min(crelu.inter).item()
        self.cur_max_inter = torch.max(crelu.inter).item()

    def _calc_inter_update_step(self) -> None:
        inter_config = self.update_config["inter"]
        update_step = inter_config["update_step"]
        if update_step != "auto":
            return update_step
        update_amount = (
            self.update_config["max_iters"] - self._first_inter_batch
        ) // inter_config["update_freq"]
        update_step = 1 / max(update_amount, 1)
        return update_step

    def _update_post_stats(self) -> None:
        for channel, cluster_details in self.cur_cluster_details.items():
            if cluster_details["all_zero"]:
                continue
            affinity_mat = get_affinity_mat(self.drelu_maps[:, channel])
            (
                cluster_details["same_label_affinity"],
                cluster_details["diff_label_affinity"],
            ) = get_mean_dist(cluster_details.get("clusters"), affinity_mat)
        self._reset_drelu_maps()


def add_crelu_hooks(
    model: nn.Module,
    layer_names: List[str],
    update_config: dict,
    layers_args: Optional[Dict[str, dict]] = None,
    **kwargs,
) -> Dict[str, CreluManager]:
    hooks = {}
    layers_args = {} if layers_args is None else layers_args
    for layer_name in layer_names:
        cur_layer_args = layers_args.get(layer_name, {})
        C, H, W = resnet18_8xb16_cifar100_Params().LAYER_NAME_TO_DIMS[layer_name]
        hook_instance = CreluManager(
            model, layer_name, update_config, C=C, H=H, W=W, **cur_layer_args, **kwargs
        )
        layer = get_layer(model, layer_name)
        layer.register_forward_hook(hook_instance.hook_fn)
        hooks[layer_name] = hook_instance
    return hooks
