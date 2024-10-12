from typing import Dict
import os
import numpy as np
import torch
from torch import nn

from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import Params as resnet18_8xb16_cifar100_Params

from research.clustering.clustering_utils import (plot_clustering, 
                                                  cluster_neurons, ClusterConvergenceException,
                                                  plot_drelu)
from model_handling import get_layer, set_layer
from research.clustering.crelu_block import ClusterRelu, prototype_from_clusters


class CreluManager:
    def __init__(self, model, layer_name, channel_idx, 
                 update_config: dict, C, H, W,
                 plot=False, out_dir=None, cluster_no_converge_fail=True):
        self.model = model
        self.layer_name = layer_name
        self.channel_idx = channel_idx
        self.update_config = update_config
        self.C, self.H, self.W = C, H, W
        self.plot = plot
        self.out_dir = out_dir
        self.cluster_no_converge_fail = cluster_no_converge_fail

        self.prefrence_quantile = self.update_config['cluster']['preference']['quantile_start']

        self.batch_idx = 0
        self.drelu_maps = None
        self.cur_cluster_res = self.cur_mean_drelu_maps = None
        self.cur_min_inter = self.cur_max_inter = None
        self._first_inter_batch = None
        self.batch_cluster_update_fail = {}
        
        self._init_crelu()
        self.cluster_started = False

        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def hook_fn(self, module: nn.Module, input, output):
        if not module.training and self.update_config['only_during_training']:
            return
        
        if self._should_update(self.update_config['drelu_stats']):
            self._update_drelu_maps(input, output)

        if self._should_update(self.update_config['cluster']):
            self._update_clusters()

        cluster_blocking_inter = self.update_config['inter']['await_cluster_start'] \
            and not self.cluster_started
        if self._should_update(self.update_config['inter']) and \
            not cluster_blocking_inter:
            self._update_inter()   

        if self.plot and self._should_update(self.update_config['plot']):
            self._plot()     

        self.batch_idx += 1

    def _plot(self):
        cluster_save_path = drelu_save_path = None
        if self.out_dir is not None:
            cluster_save_path = os.path.join(self.out_dir, 'clusters', 
                                             f'batch_{self.batch_idx}.png')
            os.makedirs(os.path.dirname(cluster_save_path), exist_ok=True)
            drelu_save_path = os.path.join(self.out_dir, 'drelu',
                                             f'batch_{self.batch_idx}.png')
            os.makedirs(os.path.dirname(drelu_save_path), exist_ok=True)
        title = f'batch {self.batch_idx}, '\
                            f'inter min {self.cur_min_inter:.2f}, '\
                            f'max {self.cur_max_inter:.2f}, '\
                            f'pref quantile {self.prefrence_quantile:.2f}'
        plot_clustering(self.cur_cluster_res, H=self.H, W=self.W, 
                        save_path=cluster_save_path, title=title)
        plot_drelu(self.cur_mean_drelu_maps, 
                   save_path=drelu_save_path,
                   title=title)
        

    def _should_update(self, config):
        update_freq = config['update_freq']
        batch_after_warmup = self.batch_idx - self.update_config['warmup']
        if update_freq is None or batch_after_warmup < 0:
            return False
        elif batch_after_warmup == 0:
            return config['update_on_start']
        return batch_after_warmup % config['update_freq'] == 0


    def _update_drelu_maps(self, input, output):
        # channel_output = output[:, self.channel_idx:self.channel_idx+1, :, :]
        channel_input = input[0][:, self.channel_idx:self.channel_idx+1]
        cur_drelu_map = channel_input.gt(0).cpu().detach()
        if self.drelu_maps is None:
            self.drelu_maps = cur_drelu_map
        else:
            self.drelu_maps = torch.concat([self.drelu_maps, cur_drelu_map], dim=0)
        self.cur_mean_drelu_maps = torch.mean(self.drelu_maps.float(), axis=0).numpy()
    
    def _update_clusters(self, reset_drelu_maps=True):
        if not self.cluster_started:
            self.cluster_started = True
        try:
            self.cur_cluster_res = cluster_neurons(self.drelu_maps, 
                                                   preference_quantile=self.prefrence_quantile,
                                                   no_converge_fail=self.cluster_no_converge_fail)
            prototype = prototype_from_clusters(self.C, self.H, self.W, {self.channel_idx: self.cur_cluster_res})
            crelu = get_layer(self.model, self.layer_name)
            crelu.prototype = prototype
        except ClusterConvergenceException as e:
            print(f"Caught convergence warning at batch {self.batch_idx}: {e}, \n"\
                      "not updating clusters")
            self.batch_cluster_update_fail[self.batch_idx] = e

        if reset_drelu_maps:
            self.drelu_maps = None
        if self.prefrence_quantile is not None:
            pref_config = self.update_config['cluster']['preference']
            self.prefrence_quantile = max(self.prefrence_quantile * pref_config['quantile_decay'],
                                           pref_config['quantile_min'])
        
    def _update_inter(self):
        if self._first_inter_batch is None:
            self._first_inter_batch = self.batch_idx
        crelu = get_layer(self.model, self.layer_name)
        update_step = self._calc_inter_update_step()
        new_inter = crelu.inter + update_step
        new_inter = torch.clamp(new_inter, 0, 1)
        crelu.inter = new_inter
        self._update_inter_stats(crelu)

    def _init_crelu(self):
        inter = self.update_config['inter'].get('default_start', 0)
        crelu = ClusterRelu(C=self.C, H=self.H, W=self.W, inter=inter)
        set_layer(self.model, self.layer_name, crelu)
        self._update_inter_stats(crelu)

    def _update_inter_stats(self, crelu):
        self.cur_min_inter = torch.min(crelu.inter).item()
        self.cur_max_inter = torch.max(crelu.inter).item()

    def _calc_inter_update_step(self):
        inter_config = self.update_config['inter']
        update_step = inter_config['update_step']
        if update_step != 'auto':
            return update_step
        update_amount = (self.update_config['max_iters'] - self._first_inter_batch) \
                // inter_config['update_freq']
        update_step = 1 / max(update_amount, 1)
        return update_step



def add_crelu_hooks(model, layer_names, channel_idx,
                    update_config, **kwargs) -> Dict[str, CreluManager]:
    hooks = {}
    for layer_name in layer_names:
        C, H, W = resnet18_8xb16_cifar100_Params().LAYER_NAME_TO_DIMS[layer_name]
        # layer = dict(model.named_modules())[layer_name]
        hook_instance = CreluManager(model, layer_name, channel_idx, update_config,
                                     C=C, H=H, W=W, **kwargs)
        layer = get_layer(model, layer_name)
        layer.register_forward_hook(hook_instance.hook_fn)
        hooks[layer_name] = hook_instance
    return hooks