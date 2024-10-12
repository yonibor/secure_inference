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

from research.clustering.crelu_manager import add_crelu_hooks


def run_train(work_dir, max_iters, validate):
    config_path = '/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py'
    ckpt = '/workspaces/secure_inference/tests/resnet18_10_8/latest.pth'
    device = 'cuda'
    seed = 42
    deterministic = False

    cfg = Config.fromfile(config_path)

    cfg.data.samples_per_gpu = 256
    cfg.optimizer.lr *= (0.2 ** 3)
    cfg.load_from = ckpt
    # cfg.runner.max_epochs = 3
    cfg.runner.max_iters = max_iters
    cfg.runner.type = 'IterBasedRunner'
    if 'max_epochs' in cfg.runner:
        del cfg.runner['max_epochs']
    cfg.lr_config = None
    cfg.workflow[0] = ('train', 5)
    cfg.workflow[1] = ('val', 0)
    cfg.evaluation = dict(interval=12, by_epoch=False)

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

    hooks = _add_crelu_hooks(model, cfg)
    torch.autograd.set_detect_anomaly(True)

    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=validate,
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)
    a = 1

    # test(model, cfg)
    # test2(model, cfg)

def _add_crelu_hooks(model, cfg):
    plot = True
    update_on_start = False
    cluster_update_freq = 4
    warmup = 12
    # warmup = 0
    only_during_training = True
    cluster_no_converge_fail = True
    

    conf = {
        'max_iters': cfg.runner.max_iters,
        'warmup': warmup,
        'only_during_training': only_during_training,
        'cluster': {
            'update_freq': cluster_update_freq, 
            'update_on_start': update_on_start,
            'preference': {
                'quantile_start': 0.5,
                'quantile_decay': 1,
                'quantile_min': 0.1,

            },
        },
        'inter': {
            'update_freq': 1, 
            'update_on_start': update_on_start,
            # 'update_step': 1 / (cfg.runner['max_iters'] - warmup),
            'update_step' : 'auto',
            #   'update_step': 1,
            # 'update_step': 0,  # TODO: remove
            'await_cluster_start': True,
        },
        'drelu_stats': {
            'update_freq': 1,
            'update_on_start': True,
        },
        'plot': {
            'update_freq': cluster_update_freq,
            'update_on_start': update_on_start,
        }
        
    }


    layers_for_hook = [
        'layer1_0_1',
    ]
    out_dir = osp.join(cfg.work_dir, 'crelu_res')

    hooks = add_crelu_hooks(model, layers_for_hook, channel_idx=0,
                            update_config=conf, plot=plot, out_dir=out_dir,
                            cluster_no_converge_fail=cluster_no_converge_fail)
    return hooks

    
def test(model, cfg):
    # ckpt = '/workspaces/secure_inference/tests/23_9_single_channel_flow/epoch_3.pth'
    ckpt = cfg.load_from
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


def test2(model, cfg):
    # config_path = '/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py'
    # cfg = Config.fromfile(config_path)
    ckpt = '/workspaces/secure_inference/tests/resnet18_10_8/latest.pth'
    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    default_args=dict(test_mode=False)
    dataset = build_dataset(cfg.data.train, default_args=default_args)

    ################ loop loader_cfg
    loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=1,
            dist=False,
            round_up=True,
            shuffle=True,  # Not shuffle by default
            sampler_cfg= None,  # Not use sampler by default
            **cfg.data.get('train_dataloader', {})
        )
    loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })
    if 'distortion_extraction' in loader_cfg:
        del loader_cfg['distortion_extraction']
    

    loader_cfg['samples_per_gpu'] = 256
    data_loader = build_dataloader(dataset, **loader_cfg)
    for i, data in enumerate(data_loader):
        if i % 4 == 0:
            print(f"Processing batch {i}")
        out = model.forward_test(data['img'])
        if i+1 >= 33:  # Stop after N examples
            break


def main():
    work_dir = '/workspaces/secure_inference/tests/25_9_single_channel_train_fixed'
    max_iters = 49
    validate = True
    run_train(work_dir, max_iters=max_iters, validate=validate)


if __name__ == '__main__':
    main()