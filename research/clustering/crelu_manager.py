from typing import Dict
import torch

from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import Params as resnet18_8xb16_cifar100_Params

from clustering.clustering_utils import plot_clustering, cluster_neurons, ClusterConvergenceException
from clustering.model_handling import get_layer, set_layer
from clustering.crelu_block import ClusterRelu, prototype_from_clusters


class CreluManager:
    def __init__(self, model, layer_name, channel_idx, 
                 update_config: dict, C, H, W,
                 plot=False):
        self.model = model
        self.layer_name = layer_name
        self.channel_idx = channel_idx
        self.update_config = update_config
        self.C, self.H, self.W = C, H, W
        self.plot = plot

        self.prefrence_quantile = self.update_config['cluster']['preference_quantile_start']

        self.batch_idx = 0
        self.drelu_maps = self.cur_cluster_res = None
        self.batch_cluster_update_fail = {}
        
        self._init_crelu()
        self.cluster_started = False

    def hook_fn(self, module, input, output):
        if self._should_update(self.update_config['drelu_stats']):
            self._update_drelu_maps(output)
        
        if self._should_update(self.update_config['cluster']):
            self._update_clusters()
            if self.plot:
                plot_clustering(self.cur_cluster_res, H=self.H, W=self.W)
            
        cluster_blocking_inter = self.update_config['inter']['await_cluster_start'] \
            and not self.cluster_started
        if self._should_update(self.update_config['inter']) and \
            not cluster_blocking_inter:
            self._update_inter()
    
        self.batch_idx += 1

    def _should_update(self, config):
        update_freq = config['update_freq']
        batch_after_warmup = self.batch_idx - config['warmup']
        if update_freq is None:
            return False
        elif batch_after_warmup == 0:
            return config['update_on_start']
        return batch_after_warmup % config['update_freq'] == 0


    def _update_drelu_maps(self, output):
        channel_output = output[:, self.channel_idx:self.channel_idx+1, :, :]
        cur_drelu_map = (channel_output != 0).cpu()
        if self.drelu_maps is None:
            self.drelu_maps = cur_drelu_map
        else:
            self.drelu_maps = torch.concat([self.drelu_maps, cur_drelu_map], dim=0)
    
    def _update_clusters(self, reset_drelu_maps=True):
        if not self.cluster_started:
            self.cluster_started = True
        try:
            self.cur_cluster_res = cluster_neurons(self.drelu_maps, 
                                                   preference_quantile=self.prefrence_quantile)
            prototype = prototype_from_clusters(self.C, self.H, self.W, {self.channel_idx: self.cur_cluster_res})
            crelu = get_layer(self.model, self.layer_name)
            crelu.prototype = prototype
        except ClusterConvergenceException as e:
            print(f"Caught convergence warning: {e}, \n"\
                      "not updating clusters")
            self.batch_cluster_update_fail[self.batch_idx] = e

        if reset_drelu_maps:
            self.drelu_maps = None
        if self.prefrence_quantile is not None:
            preference_decay = self.update_config['cluster']['preference_quantile_decay']
            self.prefrence_quantile *= preference_decay
        

    def _update_inter(self):
        crelu = get_layer(self.model, self.layer_name)
        new_inter = crelu.inter + self.update_config['inter']['update_step']
        new_inter = torch.clamp(new_inter, 0, 1)
        crelu.inter = new_inter

    def _init_crelu(self):
        inter = self.update_config['inter'].get('default_start', 0)
        crelu = ClusterRelu(C=self.C, H=self.H, W=self.W, inter=inter)
        set_layer(self.model, self.layer_name, crelu)


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