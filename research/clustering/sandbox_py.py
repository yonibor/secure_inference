import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, single_gpu_test, train_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (
    auto_select_device,
    collect_env,
    get_root_logger,
    setup_multi_processes,
    wrap_non_distributed_model,
)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import wandb
from research.clustering.crelu_logger import CreluLogger
from research.clustering.crelu_manager import add_crelu_hooks
from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2


def run_train(work_dir, max_iters, warmup, cooldown, validate, eval_interval, plot):
    wandb.login(key="8b56dc84c3fadaca2c8e6bd08ad7fc57d24c2225")

    config_path = "/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py"
    ckpt = "/workspaces/secure_inference/tests/resnet18_10_8/latest.pth"
    device = "cuda"
    seed = 42
    deterministic = False

    cfg = Config.fromfile(config_path)

    cfg.data.samples_per_gpu = 256
    cfg.optimizer.lr *= 0.2**3
    cfg.load_from = ckpt
    # cfg.runner.max_epochs = 3
    cfg.runner.max_iters = max_iters
    cfg.runner.type = "IterBasedRunner"
    if "max_epochs" in cfg.runner:
        del cfg.runner["max_epochs"]
    cfg.lr_config = None
    cfg.workflow[0] = ("train", 5)
    cfg.workflow[1] = ("val", 0)
    cfg.evaluation = dict(interval=eval_interval, by_epoch=False)

    cfg.gpu_ids = range(1)
    cfg.work_dir = work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta = dict()
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    cfg.device = device
    seed = init_random_seed(seed, device=cfg.device)
    logger.info(f"Set random seed to {seed}, " f"deterministic: {deterministic}")
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta["seed"] = seed

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
            CLASSES=datasets[0].CLASSES,
        )
    )

    if "distortion_extraction" in cfg.data:
        del cfg.data["distortion_extraction"]

    hooks = _add_crelu_hooks(model, cfg, warmup, cooldown)
    folder_name = os.path.basename(work_dir.rstrip("/"))
    cfg.log_config.hooks.append(
        dict(
            type="CreluLogger",
            output_dir=work_dir,
            crelu_hooks=hooks,
            plot=plot,
            init_kwargs={"project": "private_inference", "name": folder_name},
            interval=10,
            by_epoch=False,
        )
    )
    # torch.autograd.set_detect_anomaly(True)

    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=validate,
        timestamp=timestamp,
        device=cfg.device,
        meta=meta,
    )
    a = 1

    # test(model, cfg)
    # test2(model, cfg)


def _add_crelu_hooks(model, cfg, warmup, cooldown):
    update_on_start = False
    cluster_update_freq = 20
    only_during_training = True
    cluster_no_converge_fail = True

    conf = {
        "max_iters": cfg.runner.max_iters - cooldown,
        "warmup": warmup,
        "only_during_training": only_during_training,
        "cluster": {
            "update_freq": cluster_update_freq,
            "update_on_start": update_on_start,
            "preference": {
                "quantile_start": 0.5,
                "quantile_decay": 1,
                "quantile_min": 0.1,
            },
        },
        "inter": {
            "update_freq": 1,
            "update_on_start": update_on_start,
            # 'update_step': 1 / (cfg.runner['max_iters'] - warmup),
            "update_step": "auto",
            #   'update_step': 1,
            # 'update_step': 0,  # TODO: remove
            "await_cluster_start": True,
        },
        "drelu_stats": {
            "batch_amount": 4,
        },
        "plot": {
            "update_freq": cluster_update_freq,
            "update_on_start": update_on_start,
        },
    }

    layers_for_hook = [
        "layer1_0_1",
    ]
    output_dir = osp.join(cfg.work_dir, "crelu_res")

    hooks = add_crelu_hooks(
        model,
        layers_for_hook,
        update_config=conf,
        output_dir=output_dir,
        cluster_no_converge_fail=cluster_no_converge_fail,
    )
    return hooks


def test2(model, cfg):
    # config_path = '/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py'
    # cfg = Config.fromfile(config_path)
    ckpt = "/workspaces/secure_inference/tests/resnet18_10_8/latest.pth"
    checkpoint = load_checkpoint(model, ckpt, map_location="cpu")
    default_args = dict(test_mode=False)
    dataset = build_dataset(cfg.data.train, default_args=default_args)

    ################ loop loader_cfg
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        round_up=True,
        shuffle=True,  # Not shuffle by default
        sampler_cfg=None,  # Not use sampler by default
        **cfg.data.get("train_dataloader", {}),
    )
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k
            not in [
                "train",
                "val",
                "test",
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
            ]
        }
    )
    if "distortion_extraction" in loader_cfg:
        del loader_cfg["distortion_extraction"]

    loader_cfg["samples_per_gpu"] = 256
    data_loader = build_dataloader(dataset, **loader_cfg)
    for i, data in enumerate(data_loader):
        if i % 4 == 0:
            print(f"Processing batch {i}")
        out = model.forward_test(data["img"])
        if i + 1 >= 9:  # Stop after N examples
            break


def main():
    work_dir = "/workspaces/secure_inference/tests/2_11_single_layer_logger_debug"
    warmup = 12
    cooldown = 300
    max_iters = warmup + cooldown + 200
    eval_interval = 15
    validate = True
    plot = True

    run_train(
        work_dir,
        max_iters=max_iters,
        warmup=warmup,
        cooldown=cooldown,
        validate=validate,
        eval_interval=eval_interval,
        plot=plot,
    )


if __name__ == "__main__":
    main()
