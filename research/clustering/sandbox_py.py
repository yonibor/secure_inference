import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model, single_gpu_test
from mmcls.datasets import build_dataset, build_dataloader
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes, wrap_non_distributed_model)

from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2

from clustering.crelu_manager import add_crelu_hooks


def main():
    run_train()


def run_train():
    config_path = '/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py'
    ckpt = '/workspaces/secure_inference/tests/resnet18_10_8/latest.pth'
    work_dir = '/workspaces/secure_inference/tests/23_9_single_channel_flow'
    device = 'cuda'
    seed = 42
    deterministic = False

    cfg = Config.fromfile(config_path)

    cfg.data.samples_per_gpu = 256
    cfg.optimizer.lr *= (0.2 ** 3)
    cfg.load_from = ckpt
    # cfg.runner.max_epochs = 3
    cfg.runner.max_iters = 100
    cfg.runner.type = 'IterBasedRunner'
    if 'max_epochs' in cfg.runner:
        del cfg.runner['max_epochs']
    cfg.lr_config = None

    cfg.gpu_ids = range(1)
    cfg.work_dir = work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta = dict()
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
    cfg.device = device
    seed = init_random_seed(seed, device=cfg.device)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {deterministic}')
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # save mmcls version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES))
    
    if 'distortion_extraction' in cfg.data:
        del cfg.data['distortion_extraction']

    hooks = _add_crelu_hooks(model)

    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)

    # test(model, cfg)

def _add_crelu_hooks(model):
    plot = True
    update_on_start = True
    cluster_update_freq = 1
    warmup = 20
    

    conf = {
        'cluster': {'update_freq': cluster_update_freq, 
                    'update_on_start': update_on_start,
                    'preference_quantile_start': None,
                    'preference_quantile_decay': None,
                    'warmup': warmup},
        'inter': {'update_freq': 1, 
                'update_on_start': update_on_start,
                'update_step': 1 / (model.runner['max_iter'] - warmup),
                #   'update_step': 1,
                'await_cluster_start': True,
                'warmup': warmup},
        'drelu_stats': {'update_freq': cluster_update_freq,
                        'update_on_start': update_on_start,
                        'warmup': warmup}
    }


    layers_for_hook = [
        'layer1_0_1',
    ]

    hooks = add_crelu_hooks(model, layers_for_hook, channel_idx=0,
                            update_config=conf, plot=plot)
    return hooks

    
def test(model, cfg):
    ckpt = '/workspaces/secure_inference/tests/23_9_single_channel_flow/epoch_3.pth'
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=False,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    checkpoint = load_checkpoint(model, ckpt, cfg.device)
    model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
    
    outputs = single_gpu_test(model, data_loader)
    eval_results = dataset.evaluate(results=outputs)
    print(eval_results)
    



if __name__ == '__main__':
    main()