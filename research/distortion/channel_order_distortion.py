import argparse
import copy
import os

from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import mmcv

from research.distortion.parameters.factory import param_factory
from research.distortion.distortion_utils import DistortionUtils
from research.distortion.utils import get_channels_subset
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2  # TODO: why is this needed?
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.mmlab_extension.transforms import CenterCrop

class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path, params, cfg, is_train_mode=False):

        self.params = params
        self.cfg = cfg
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params, cfg=self.cfg, mode="distortion_extraction")
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

        if is_train_mode:
            self.distortion_utils.model.train()

    def extract_deformation_channel_ord(self,
                                        batch_index: int,
                                        layer_names: List[str],
                                        batch_size: int,
                                        baseline_block_size_spec: Dict[str, np.array],
                                        clean_block_size_spec: Dict[str, np.array],
                                        seed: int,
                                        cur_iter: int,
                                        num_iters: int):

        channels_to_run, _ = get_channels_subset(seed=seed, params=self.params, cur_iter=cur_iter, num_iters=num_iters)

        os.makedirs(self.output_path, exist_ok=True)

        for layer_name in layer_names:
            file_name = os.path.join(self.output_path, f"{layer_name}_{batch_index}.pickle")
            if os.path.exists(file_name):
                continue
            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
            input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            output_block_name = self.params.BLOCK_NAMES[-2]  # TODO: Why do we have None in last layer

            layer_assets = {key: np.zeros(shape=(layer_num_channels, len(block_sizes))) for key in self.keys}

            block_size_spec = copy.deepcopy(baseline_block_size_spec)

            if layer_name not in block_size_spec:
                block_size_spec[layer_name] = np.ones(shape=(layer_num_channels, 2), dtype=np.int32)

            for channel in tqdm(channels_to_run[layer_name], desc=f"Batch={batch_index} Layer={layer_name}"):
                for block_size_index, block_size in enumerate(block_sizes):

                    orig_block_size = block_size_spec[layer_name][channel].copy()
                    block_size_spec[layer_name][channel] = block_size

                    cur_assets = self.distortion_utils.get_batch_distortion(
                        clean_block_size_spec=clean_block_size_spec,
                        baseline_block_size_spec=baseline_block_size_spec,
                        block_size_spec=block_size_spec,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        input_block_name=input_block_name,
                        output_block_name=output_block_name)

                    block_size_spec[layer_name][channel] = orig_block_size

                    for key in self.keys:
                        layer_assets[key][channel, block_size_index] = cur_assets[key]

            with open(file_name, "wb") as file:
                pickle.dump(obj=layer_assets, file=file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--batch_index', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py")
    parser.add_argument('--checkpoint', type=str, default="/home/yakir/epoch_14.pth")
    parser.add_argument('--baseline_block_size_spec', type=str, default="/home/yakir/4x4.pickle")
    parser.add_argument('--clean_block_size_spec', type=str, default=None)
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/tmp_12/channel_distortions")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cur_iter', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--train_mode', action='store_true', default=False)

    args = parser.parse_args()
    seed = None

    cfg = mmcv.Config.fromfile(args.config)
    gpu_id = args.gpu_id
    params = param_factory(cfg)

    params.CHECKPOINT = args.checkpoint

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    layer_names = params.LAYER_NAMES[::-1]

    if args.baseline_block_size_spec and os.path.exists(args.baseline_block_size_spec):
        baseline_block_size_spec = pickle.load(open(args.baseline_block_size_spec, 'rb'))
    else:
        baseline_block_size_spec = dict()

    if args.clean_block_size_spec and os.path.exists(args.clean_block_size_spec):
        clean_block_size_spec = pickle.load(open(args.clean_block_size_spec, 'rb'))
    else:
        clean_block_size_spec = dict()

    chd = ChannelDistortionHandler(gpu_id=gpu_id,
                                   output_path=output_path,
                                   params=params,
                                   cfg=cfg,
                                   is_train_mode=args.train_mode)

    chd.extract_deformation_channel_ord(batch_index=args.batch_index,
                                        layer_names=layer_names,
                                        batch_size=args.batch_size,
                                        baseline_block_size_spec=baseline_block_size_spec,
                                        clean_block_size_spec=clean_block_size_spec,
                                        seed=seed,
                                        cur_iter=args.cur_iter,
                                        num_iters=args.num_iters)
