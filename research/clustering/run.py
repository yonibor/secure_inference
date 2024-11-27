import argparse
import copy
import os
import os.path as osp
import time
import warnings
from copy import deepcopy

import mmcv
import numpy as np
import pandas as pd
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
from research.clustering.crelu_manager import CreluManager, add_crelu_hooks
from research.clustering.model.resnet_cifar_crelu import (
    ResNet_CIFAR_CRELU,  # import for the register
)
from research.clustering.prioritize_by_knapsack import (
    format_per_layer,
    prioritize_channels,
)
from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import (
    Params,
)

# from research.mmlab_extension.classification.resnet_cifar_v2 import (
#     ResNet_CIFAR_V2,  # import for the register
# )


def run_train(work_dir, validate, eval_interval, plot, batch_size, ckpt, hooks_kwargs):
    wandb.login(key="8b56dc84c3fadaca2c8e6bd08ad7fc57d24c2225")

    config_path = "/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py"
    device = "cuda"
    seed = 42
    deterministic = False

    cfg = Config.fromfile(config_path)

    cfg["model"]["backbone"]["type"] = "ResNet_CIFAR_CRELU"
    cfg["model"]["pretrained"] = ckpt

    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 0
    cfg.data.persistent_workers = False

    cfg.optimizer.lr *= 0.2**3

    # cfg.runner.max_epochs = 3
    cfg.runner.type = "IterBasedRunner"
    if "max_epochs" in cfg.runner:
        del cfg.runner["max_epochs"]
    cfg.lr_config = {"policy": "step", "step": [5000, 25000], "gamma": 0.2}
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

    hooks, max_iters, cluster_end = _add_crelu_hooks(
        model, work_dir, batch_size=batch_size, **hooks_kwargs
    )
    cfg.checkpoint_config["interval"] = cluster_end + 100
    cfg.runner.max_iters = max_iters
    folder_name = os.path.basename(work_dir.rstrip("/"))
    parent_folder_name = os.path.normpath(work_dir).split(os.sep)[-2]
    cfg.log_config.hooks.append(
        dict(
            type="CreluLogger",
            output_dir=work_dir,
            crelu_hooks=hooks,
            plot=plot,
            init_kwargs={
                "project": "private_inference",
                "name": folder_name,
                "group": parent_folder_name,
                "config": {
                    **hooks_kwargs,
                    "lr": cfg.optimizer.lr,
                    "max_iters": max_iters,
                    "batch_size": batch_size,
                },
            },
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


def _add_crelu_hooks(
    model,
    work_dir,
    warmup,
    cluster_cooldown,
    cluster_update_freq,
    drelu_stats_batch_amount,
    inter_config,
    cluster_once,
    # preference_start,
    group_channels_config,
    sigmoid_config,
    id_config,
    clustering_iters,
    batch_size,
    clustering_stats_dir,
    knapsack_path,
    layer_names,
    **kwargs,
):
    update_on_start = True
    only_during_training = True
    cluster_no_converge_fail = True
    # layers_args = {"layer1_0_1": {"keep_channels": [47, 24, 9, 12, 29]}}
    layers_args = {}
    use_cluster_mean = False

    priority = prioritize_channels(
        clustering_stats_dir, knapsack_path, Params().LAYER_NAME_TO_DIMS
    )
    os.makedirs(work_dir, exist_ok=True)
    priority.to_csv(os.path.join(work_dir, "prioritize"))
    # priority = pd.read_csv(os.path.join(work_dir, "prioritize"))

    formatted_priority = format_per_layer(priority, layer_names)

    config = {
        "general": {
            "only_during_training": only_during_training,
        },
        "warmup": {
            "iters": warmup,
        },
        "id_warmup": id_config,
        "cluster": {
            "update_freq": cluster_update_freq,
            "update_on_start": update_on_start,
            "cluster_once": cluster_once,
            "iters": clustering_iters,
            # "preference": {
            #     "quantile_start": preference_start,
            #     "quantile_decay": 1,
            #     "quantile_min": 0.1,
            # },
            "cooldown": cluster_cooldown,
            "use": True,
        },
        "inter": inter_config,
        "drelu_stats": {
            "batch_amount": drelu_stats_batch_amount,
        },
        "plot": {
            "update_freq": cluster_update_freq,
            "update_on_start": update_on_start,
        },
        "sigmoid": sigmoid_config,
        "group_channels": group_channels_config,
    }

    for layer_name in layer_names:
        layer_config = {"config": deepcopy(config)}
        layer_config.update(formatted_priority[layer_name])
        layers_args[layer_name] = layer_config

    # layers_for_hook = [
    #     # "layer1_0_1",
    #     "layer1_0_2",
    # ]
    # layers_for_hook = Params().LAYER_NAMES

    hooks = add_crelu_hooks(
        model=model,
        # update_config=config,
        cluster_no_converge_fail=cluster_no_converge_fail,
        layers_args=layers_args,
        use_cluster_mean=use_cluster_mean,
        batch_size=batch_size,
        layer_names=layer_names,
        **kwargs,
    )

    max_iters, cluster_end = CreluManager.get_iters_end(config)
    return hooks, max_iters, cluster_end


# def test2(model, cfg):
#     # config_path = '/workspaces/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py'
#     # cfg = Config.fromfile(config_path)
#     ckpt = "/workspaces/secure_inference/tests/resnet18_10_8/latest.pth"
#     checkpoint = load_checkpoint(model, ckpt, map_location="cpu")
#     default_args = dict(test_mode=False)
#     dataset = build_dataset(cfg.data.train, default_args=default_args)

#     ################ loop loader_cfg
#     loader_cfg = dict(
#         # cfg.gpus will be ignored if distributed
#         num_gpus=1,
#         dist=False,
#         round_up=True,
#         shuffle=True,  # Not shuffle by default
#         sampler_cfg=None,  # Not use sampler by default
#         **cfg.data.get("train_dataloader", {}),
#     )
#     loader_cfg.update(
#         {
#             k: v
#             for k, v in cfg.data.items()
#             if k
#             not in [
#                 "train",
#                 "val",
#                 "test",
#                 "train_dataloader",
#                 "val_dataloader",
#                 "test_dataloader",
#             ]
#         }
#     )
#     if "distortion_extraction" in loader_cfg:
#         del loader_cfg["distortion_extraction"]

#     loader_cfg["samples_per_gpu"] = 256
#     data_loader = build_dataloader(dataset, **loader_cfg)
#     for i, data in enumerate(data_loader):
#         if i % 4 == 0:
#             print(f"Processing batch {i}")
#         out = model.forward_test(data["img"])
#         if i + 1 >= 9:  # Stop after N examples
#             break

# from memory_profiler import profile


# @profile
def main():
    # layers = list(
    #     set(Params().LAYER_NAMES)
    #     - set(["layer2_0_1", "layer3_0_1", "layer4_0_1", "layer3_1_2"])
    # )
    # layers_for_hook=["layer1_0_1"],
    layers = Params().LAYER_NAMES
    # layers = ["layer3_0_1"]

    group_channels_config = dict(
        group=False,
        group_by_kmeans=True,
        k=50,
        group_channels_once=True,
    )

    sigmoid_config = dict(
        use=False,
        cluster_factor=6,
        relu_factor=36,
        to_sigmoid=dict(
            iters=2000,
            cooldown=1000,
        ),
        from_sigmoid=dict(
            iters=2000,
            cooldown=4000,
        ),
    )

    id_config = {
        # "iters": 6000,
        "id_warmup": 300,
        "use": False,
    }

    warmup = 20
    # warmup = 300
    clustering_iters = 1000
    cooldown = 10000

    inter_config = dict(start_value=0, end_value=1, before_activation=True)

    ckpt = "/workspaces/secure_inference/tests/resnet18_10_8/latest.pth"
    # ckpt = "/workspaces/secure_inference/tests/24_11_prioritize/full_08_v3/latest.pth"

    # perfs = np.concatenate([np.arange(0.2, 0.8, 0.1), np.arange(0.25, 0.8, 0.1)])

    # print(f"------------------perf {perf}----------------")
    run_train(
        work_dir=f"/workspaces/secure_inference/tests/26_11_multi_prototype/all_stats",
        validate=True,
        # eval_interval=200,
        eval_interval=1500,
        plot=False,
        ckpt=ckpt,
        # layer_names=Params().LAYER_NAMES,
        hooks_kwargs=dict(
            layer_names=layers,
            cluster_update_freq=1000,
            # cluster_update_freq=8,
            warmup=warmup,
            cluster_cooldown=cooldown,
            clustering_iters=clustering_iters,
            drelu_stats_batch_amount=8,
            cluster_once=True,
            # preference_start=perf,
            use_crelu_existing_params=False,
            group_channels_config=group_channels_config,
            sigmoid_config=sigmoid_config,
            inter_config=inter_config,
            id_config=id_config,
            clustering_stats_dir="/workspaces/secure_inference/tests/22_cluster_amount_stats",
            knapsack_path="/workspaces/secure_inference/tests/distortion_extraction_22_11_512/block_spec/0.08.pickle",
        ),
        batch_size=128,
    )


if __name__ == "__main__":
    main()
