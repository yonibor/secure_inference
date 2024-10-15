import os
import csv
from typing import Dict
import numpy as np

from mmcv.runner.hooks.logger.wandb import WandbLoggerHook
from mmcv.runner.hooks.hook import HOOKS

from research.clustering.crelu_manager import CreluManager

@HOOKS.register_module(WandbLoggerHook)
class CreluLogger:
    def __init__(self, crelu_hooks: Dict[str, CreluManager], output_dir, **kwargs):
        super().__init__(**kwargs)
        self.crelu_hooks = crelu_hooks
        self.output_dir = output_dir
        self.channel_path = os.path.join(self.output_dir, 'per_channel.csv')
        self._init_files()
        self.last_batch_idx = self._get_batch_idx()
        self._update_batch_idx()
        self.layer_name_example = sorted(list(self.crelu_hooks.keys()))[0]

    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        if self._should_update():
            self._log_general(runner)
            self._log_channels()
        self.last_batch_idx = self._get_batch_idx()

    def _get_batch_idx(self):
        return self.crelu_hooks[self.layer_name_example].batch_idx

    def _should_update(self):
        batch_idx = self._get_batch_idx()
        return (self.last_batch_idx is None and batch_idx is not None) or \
            (self.last_batch_idx < batch_idx)

    def _log_general(self, runner):
        with open(self.general_path, 'a', newline='') as file:
            tags = {}
            for layer_name, crelu in self.crelu_hooks.items():
                channels_details = self._get_channels_details(crelu)
                channels_summary = self._summarize_channels_details(channels_details)
                layer_tags = dict(
                    batch_index=self.crelu_hooks.batch_idx,
                    min_inter=crelu.cur_min_inter,
                    max_inter=crelu.cur_max_inter,
                    preference=crelu.prefrence_quantile,
                    **channels_summary
                )
                layer_tags = {f'{layer_name}_{k}': v for k, v in layer_tags}
                tags.update(layer_tags)
            tags.update(self._summarize_layers(tags))
            self._log_wandb(tags, runner)
    
    def _get_channels_details(self, crelu: CreluManager):
        res = {}
        channel_drelu_mean = crelu.cur_mean_drelu_maps.reshape(
            crelu.cur_mean_drelu_maps.shape[0], -1).mean(axis=1)
        for channel in range(crelu.C):
            cluster_details = crelu.cur_cluster_details[channel]
            if cluster_details['all_zero']:
                cluster_amount = None
            elif cluster_details['clusters'] is None:
                cluster_amount = crelu.C * crelu.H * crelu.W
            else:
                cluster_amount = cluster_details['clusters'].cluster_centers_indices_

            clusters = cluster_details['clusters']
            cluster_amount = None if clusters is None else \
                    len()
            copy_cols = ['all_zero', 'failed_to_converge',
                         'same_label_affinity', 'diff_label_affinity']
            res[channel] = {
                **{k: v for k, v in cluster_details.items if k in copy_cols},
                'cluster_amount': cluster_amount,
                'drelu_mean': channel_drelu_mean[channel]
            }
        return res
    
    def _summarize_channels_details(channels_details, crelu: CreluManager):
        def _sum_key(k):
            return sum([float(details[k]) for details in 
                        channels_details.values() if details[k] is not None])
        
        details = dict(
            all_zero_ratio=crelu.C/_sum_key('all_zero'),
            failed_ratio=crelu.C/_sum_key('failed_to_converge'),
            cluster_amount_mean=_sum_key('cluster_amount')/crelu.C
        )
        return details
        

    def _summarize_layers(self, tags):
        original_drelu_amount = cur_drelu_amount = 0
        channel_amount = all_zero_amount = fail_amount = 0
        for layer_name, crelu in self.crelu_hooks.items():
            original_drelu_amount += crelu.C * crelu.W * crelu.H
            cluster_mean_amount = tags[f'{layer_name}_cluster_amount_mean']
            all_zero_ratio = tags[f'{layer_name}_all_zero_ratio']
            cur_drelu_amount += crelu.C * (1 - all_zero_ratio) * cluster_mean_amount
            channel_amount += crelu.C
            all_zero_amount += all_zero_ratio * crelu.C
            fail_amount += tags[f'{layer_name}_failed_ratio'] * crelu.C
        details = dict(
            total_drelu_ratio=cur_drelu_amount / original_drelu_amount,
            total_all_zero_ratio=all_zero_amount/channel_amount,
            total_fail_amount=fail_amount/channel_amount
        )
        return details
            
    def _log_wandb(self, tags, runner):
        if self.with_step:
            self.wandb.log(
                tags, step=self.get_iter(runner), commit=self.commit)
        else:
            tags['global_step'] = self.get_iter(runner)
            self.wandb.log(tags, commit=self.commit)

                
    def _log_channels(self):
        with open(self.channel_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for layer_name, crelu in self.crelu_hooks.items():
                if (not crelu.should_update_clusters() and not crelu.should_update_post_stats()):
                    return
                channels_details = self._get_channels_details(crelu)
                for channel, details in channels_details.items():
                    writer.writerow([crelu.batch_idx, layer_name, channel,
                                    details['is_active'], 
                                    details['fail'], 
                                    details['cluster_amount'],
                                    details['same_label_affinity'],
                                    details['diff_label_affinity'],
                                    details['drelu_mean']])


    def _init_files(self):
        os.makedirs(os.path.dirname(self.general_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.channel_path), exist_ok=True)
        general_cols = ['batch_index', 'layer_name', 'min_inter', 'max_inter', 'prefrence']
        channel_cols = ['batch_index', 'layer_name', 'channel_index', 'is_active', 
                        'failed_convergence', 'cluster_amount', 
                        'same_label_affinity', 'diff_label_affinity',
                        'mean_drelu']
        for path, cols in [(self.general_path, general_cols), 
                           (self.channel_path, channel_cols)]:
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(cols)


