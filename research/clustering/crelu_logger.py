import os
import csv
from typing import Dict

class CreluLogger:
    def __init__(self, hook, output_dir):
        self.hook = hook
        self.layer_name = hook.layer_name
        self.output_dir = output_dir
        self.general_path = os.path.join(self.output_dir, self.layer_name, 'general.csv')
        self.channel_path = os.path.join(self.output_dir, self.layer_name, 'per_channel.csv')
        self._init_files()

    def after_train_iter(self, runner=None):
        self._log_general()
        self._log_channels()

    def _log_general(self):
        with open(self.general_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.hook.batch_idx, self.layer_name, 
                             self.hook.cur_min_inter,
                             self.hook.cur_max_inter, self.hook.prefrence_quantile])
                
    def _log_channels(self):
        with open(self.channel_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not self.hook.should_update_clusters():
                return
            channel_drelu_mean = self.hook.cur_mean_drelu_maps.reshape(self.hook.cur_mean_drelu_maps.shape[0], -1).mean(axis=1)
            for channel_idx in range(self.hook.C):
                cluster_details = self.hook.cur_cluster_res[channel_idx]
                is_active = not cluster_details['all_zero']
                fail = cluster_details['failed_to_converge']
                cluster_res = cluster_details['cluster_res']
                cluster_amount = None if cluster_res is None else \
                    len(cluster_res.cluster_centers_indices_) 
                writer.writerow([self.hook.batch_idx, self.layer_name, channel_idx,
                                 is_active, fail, cluster_amount,
                                 cluster_details['same_label_affinity'],
                                 cluster_details['diff_label_affinity'],
                                 channel_drelu_mean[channel_idx]])


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


