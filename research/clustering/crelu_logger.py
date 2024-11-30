import csv
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

from research.clustering.clustering_utils import plot_clustering, plot_drelu
from research.clustering.crelu_manager import CreluManager

CHANNEL_COLS = [
    "batch_index",
    "layer_name",
    "channel",
    "id",
    "channels",
    "all_zero",
    "failed_to_converge",
    "cluster_amount",
    "same_label_affinity",
    "diff_label_affinity",
    "prototype_affinity",
    "drelu_mean",
    "preference",
    "id_channel",
    "keep_relu_channel",
    "decisions_accuracy",
    "decisions_id_accuracy",
    "decisions_zero_accuracy",
]


@HOOKS.register_module
class CreluLogger(WandbLoggerHook):
    def __init__(
        self,
        crelu_hooks: Dict[str, CreluManager],
        output_dir: str,
        plot: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(commit=False, **kwargs)
        self.crelu_hooks = crelu_hooks

        self.output_dir = output_dir
        self.plot = plot
        self.channel_path = os.path.join(self.output_dir, "per_channel.csv")
        self.cluster_dir = os.path.join(self.output_dir, "cluster_plots")
        self.drelu_dir = os.path.join(self.output_dir, "drelu_plots")
        self._init_files()

        layer_name_example = sorted(list(self.crelu_hooks.keys()))[0]
        self.crelu_example = self.crelu_hooks[layer_name_example]
        self.last_batch_idx = self.crelu_example.batch_idx
        self.already_plotted = {layer_name: False for layer_name in self.crelu_hooks}

    def after_train_iter(self, runner) -> None:
        super().after_train_iter(runner)
        if self._should_update():
            self._log_general(runner)
            self._log_channels()
            if self.plot and self.crelu_example.should_update_clusters():
                self._plot()
        self.last_batch_idx = self.crelu_example.batch_idx

    def _should_update(self) -> bool:
        cur_batch_idx = self.crelu_example.batch_idx
        batch_changed = (self.last_batch_idx is None and cur_batch_idx is not None) or (
            self.last_batch_idx < cur_batch_idx
        )
        # return batch_changed and self.crelu_example.cluster_started
        return batch_changed

    def _log_general(self, runner) -> None:
        tags = {}
        for layer_name, crelu_manager in self.crelu_hooks.items():
            layer_tags = dict(
                batch_index=crelu_manager.batch_idx,
                min_inter=crelu_manager.crelu.inter.min().item(),
                max_inter=crelu_manager.crelu.inter.max().item(),
                use_sigmoid=int(crelu_manager.crelu.use_sigmoid),
                sigmoid_factor=crelu_manager.crelu.sigmoid_factor,
            )
            if crelu_manager.cluster_started:
                groups_details = self._get_groups_details(
                    crelu_manager, return_per_channel=False
                )
                channels_summary = self._summarize_groups_details(groups_details)
                layer_tags.update(channels_summary)
            layer_tags = {f"{layer_name}/{k}": v for k, v in layer_tags.items()}
            tags.update(layer_tags)
        if crelu_manager.cluster_started:
            tags.update(self._summarize_layers(tags))
        self._log_wandb(tags, runner)

    def _get_groups_details(
        self, crelu_manager: CreluManager, return_per_channel: bool
    ) -> Union[Dict[str, dict], List[dict]]:
        res = []
        res_per_channel = {}
        channel_drelu_mean = crelu_manager.cur_mean_drelu_maps.reshape(
            crelu_manager.cur_mean_drelu_maps.shape[0], -1
        ).mean(axis=1)
        for clusters_details in crelu_manager.cur_clusters_details:
            clusters = clusters_details["clusters"]
            features_data = clusters_details["features_data"]
            assert (
                len(clusters_details["channels"]) == 1
            ), "currently not supported with filtering channles"
            channel = clusters_details["channels"][0]
            id_channel = channel in crelu_manager.id_channels
            decisions_accuracy = decisions_id_accuracy = decisions_zero_accuracy = None
            if clusters_details["all_zero"] or id_channel:
                cluster_amount = None
            elif clusters is None:
                cluster_amount = (
                    crelu_manager.H
                    * crelu_manager.W
                    * clusters_details["channels"].size
                )
            else:
                cluster_amount = len(clusters["centers_indices"])
                decisions_accuracy = features_data["accuracy"]
                decisions_id_accuracy = features_data["id_accuracy"]
                decisions_zero_accuracy = features_data["zero_accuracy"]

            copy_keys = [
                "all_zero",
                "failed_to_converge",
                "same_label_affinity",
                "diff_label_affinity",
                "prototype_affinity",
                "id",
                "channels",
            ]
            cur_res = {
                **{k: clusters_details[k] for k in copy_keys},
                "cluster_amount": cluster_amount,
                "id_channel": id_channel,
                "preference": crelu_manager.preference.get(channel),
                "keep_relu_channel": channel in crelu_manager.keep_relu_channels,
                "decisions_accuracy": decisions_accuracy,
                "decisions_id_accuracy": decisions_id_accuracy,
                "decisions_zero_accuracy": decisions_zero_accuracy,
            }
            res.append(cur_res)
            for channel in clusters_details["channels"]:
                channel_res = cur_res.copy()
                channel_res["drelu_mean"] = channel_drelu_mean[channel]
                res_per_channel[channel] = channel_res
        if return_per_channel:
            return res_per_channel
        else:
            return res

    @staticmethod
    def _summarize_groups_details(
        groups_details: List[dict],
    ) -> Dict[str, np.float32]:
        def _agg_key(k, mean, allow_none=False):
            details_not_none = [
                details for details in groups_details if details[k] is not None
            ]
            assert allow_none or len(details_not_none) == len(groups_details)
            results = np.array([details[k] for details in details_not_none])
            if mean:
                sizes = np.array(
                    [details["channels"].size for details in details_not_none]
                )
                results = np.sum(results * sizes) / np.sum(sizes)
            else:
                results = results.sum()
            return results

        details = dict(
            all_zero_ratio=_agg_key("all_zero", mean=True),
            id_channel_ratio=_agg_key("id_channel", mean=True),
            failed_ratio=_agg_key("failed_to_converge", mean=True),
            cluster_amount_mean=_agg_key("cluster_amount", mean=True, allow_none=True),
            cluster_amount_sum=_agg_key("cluster_amount", mean=False, allow_none=True),
            decisions_accuracy=_agg_key(
                "decisions_accuracy", mean=True, allow_none=True
            ),
            decisions_id_accuracy=_agg_key(
                "decisions_id_accuracy", mean=True, allow_none=True
            ),
            decisions_zero_accuracy=_agg_key(
                "decisions_zero_accuracy", mean=True, allow_none=True
            ),
        )
        return details

    def _summarize_layers(self, tags: dict) -> dict:
        original_drelu_amount = cur_drelu_amount = 0
        channel_amount = all_zero_amount = fail_amount = id_channel_amount = 0

        for layer_name, crelu in self.crelu_hooks.items():
            original_drelu_amount += crelu.C * crelu.W * crelu.H
            all_zero_ratio = tags[f"{layer_name}/all_zero_ratio"]
            id_channel_ratio = tags[f"{layer_name}/id_channel_ratio"]
            cur_drelu_amount += tags[f"{layer_name}/cluster_amount_sum"]
            channel_amount += crelu.C
            all_zero_amount += all_zero_ratio * crelu.C
            id_channel_amount += id_channel_ratio * crelu.C
            fail_amount += tags[f"{layer_name}/failed_ratio"] * crelu.C

        details = {
            "total/drelu_ratio": cur_drelu_amount / original_drelu_amount,
            "total/all_zero_ratio": all_zero_amount / channel_amount,
            "total/id_channel_ratio": id_channel_amount / channel_amount,
            "total/fail_ratio": fail_amount / channel_amount,
        }
        return details

    def _log_wandb(self, tags: dict, runner) -> None:
        if self.with_step:
            self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
        else:
            tags["global_step"] = self.get_iter(runner)
            self.wandb.log(tags, commit=self.commit)

    def _log_channels(self) -> None:
        with open(self.channel_path, "a", newline="") as file:
            writer = csv.writer(file)
            for layer_name, crelu in self.crelu_hooks.items():
                if (
                    not crelu.should_update_clusters()
                    and not crelu.should_update_post_stats()
                ):
                    return
                channels_details = self._get_groups_details(
                    crelu, return_per_channel=True
                )
                for channel, details in channels_details.items():
                    details.update(
                        {
                            "batch_index": crelu.batch_idx,
                            "layer_name": layer_name,
                            "channel": channel,
                        }
                    )
                    writer.writerow([details[c] for c in CHANNEL_COLS])

    def _init_files(self) -> None:
        os.makedirs(os.path.dirname(self.channel_path), exist_ok=True)
        for layer_name in self.crelu_hooks:
            os.makedirs(os.path.join(self.cluster_dir, layer_name), exist_ok=True)
            os.makedirs(os.path.join(self.drelu_dir, layer_name), exist_ok=True)
        with open(self.channel_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CHANNEL_COLS)

    def _plot(self) -> None:
        for layer_name, crelu in self.crelu_hooks.items():
            if self.already_plotted[layer_name]:
                continue
            self.already_plotted[layer_name] = True
            for clusters_details in crelu.cur_clusters_details:
                channel = clusters_details["id"]
                cluster_path = os.path.join(
                    self.cluster_dir,
                    layer_name,
                    f"batch_{crelu.batch_idx}_channel_{channel}.png",
                )
                drelu_path = os.path.join(
                    self.drelu_dir,
                    layer_name,
                    f"batch_{crelu.batch_idx}_channel_{channel}.png",
                )
                title = f"batch: {crelu.batch_idx}, channel: {channel}"
                clusters = clusters_details["clusters"]
                if clusters is not None:
                    cluster_amount = len(clusters["centers_indices"])
                    clusters_title = f"{title}, cluster amount: {cluster_amount}"
                    plot_clustering(
                        labels=clusters["labels"],
                        cluster_centers_indices=clusters["centers_indices"],
                        H=crelu.H,
                        W=crelu.W,
                        save_path=cluster_path,
                        title=clusters_title,
                    )
                plot_drelu(
                    crelu.cur_mean_drelu_maps[channel],
                    save_path=drelu_path,
                    title=title,
                )
