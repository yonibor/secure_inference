import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import mmcv

from research.distortion.parameters.factory import param_factory
from research.distortion.distortion_utils import DistortionUtils, get_block_spec_num_relus, get_num_relus
from research.distortion.utils import get_channel_order_statistics, get_num_of_channels
from research.distortion.utils import get_channels_subset
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.distortion.arch_utils.factory import arch_utils_factory
import torch
class SimulatedAnnealingHandler:
    def __init__(self, gpu_id, params, cfg, input_block_spec_path, output_block_spec_path):

        self.params = params
        self.cfg = cfg
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params, cfg=self.cfg)

        self.keys = ["Noise", "Signal"]
        self.block_size_spec = pickle.load(open(input_block_spec_path, "rb"))
        self.output_block_spec_path = output_block_spec_path
        self.channel_order_to_layer, self.channel_order_to_channel, self.channel_order_to_dim = get_channel_order_statistics(self.params)

        self.num_channels = get_num_of_channels(self.params)
        self.num_of_drelus = get_block_spec_num_relus(self.block_size_spec, self.params)

        self.dim_to_channels = {dim: np.argwhere(self.channel_order_to_dim == dim)[:,0] for dim in np.unique(self.channel_order_to_dim)}
        self.flipped = 0
        self.arch_utils = arch_utils_factory(self.cfg)
        self.distortion_utils.model.train()

    def get_sibling_channels(self):
        random_channel_a = np.random.choice(self.num_channels)
        channels_b = self.dim_to_channels[self.channel_order_to_dim[random_channel_a]]
        random_channel_b = np.random.choice(channels_b)
        return random_channel_a, random_channel_b

    def get_suggested_block_size(self, iteration):

        while True:
            suggest_block_size_spec = copy.deepcopy(self.block_size_spec)
            sibling_channel_a, sibling_channel_b = self.get_sibling_channels()
            layer_name_a = self.channel_order_to_layer[sibling_channel_a]
            layer_name_b = self.channel_order_to_layer[sibling_channel_b]
            channel_a = self.channel_order_to_channel[sibling_channel_a]
            channel_b = self.channel_order_to_channel[sibling_channel_b]

            if not np.all(suggest_block_size_spec[layer_name_a][channel_a] == suggest_block_size_spec[layer_name_b][channel_b]):

                tmp = suggest_block_size_spec[layer_name_a][channel_a]
                suggest_block_size_spec[layer_name_a][channel_a] = suggest_block_size_spec[layer_name_b][channel_b]
                suggest_block_size_spec[layer_name_b][channel_b] = tmp
                return suggest_block_size_spec

    def get_batch_size(self, iteration):
        return 512

    def get_batch_index(self, iteration):
        return 0
        batch_size = self.get_batch_size(iteration)
        return np.random.choice(len(self.distortion_utils.dataset) // batch_size)

    def extract_deformation_channel_ord(self, iteations):

        steps = []
        distorted_losses = []
        baseline_losses = []
        for iteration in range(iteations):
            torch.cuda.empty_cache()
            print(iteration)
            batch_size = self.get_batch_size(iteration)
            batch_index = self.get_batch_index(iteration)
            suggest_block_size_spec = self.get_suggested_block_size(iteration)

            np.random.shuffle(self.distortion_utils.shuffled_indices)
            batch, ground_truth = self.distortion_utils.get_samples(batch_index, batch_size)
            ground_truth = ground_truth.to(torch.long)

            self.arch_utils.set_bReLU_layers(self.distortion_utils.model, self.block_size_spec)
            baseline_loss = float(self.distortion_utils.model(batch, gt_label=ground_truth, return_loss=True)['loss'].detach().cpu().numpy())

            self.arch_utils.set_bReLU_layers(self.distortion_utils.model, suggest_block_size_spec)
            distorted_loss = float(self.distortion_utils.model(batch, gt_label=ground_truth, return_loss=True)['loss'].detach().cpu().numpy())

            if distorted_loss < baseline_loss:
                steps.append(iteration)
                distorted_losses.append(distorted_loss)
                baseline_losses.append(baseline_loss)
                self.flipped += 1
                self.block_size_spec = suggest_block_size_spec

                if self.flipped % 100 == 1:
                    pickle.dump(obj=self.block_size_spec, file=open(self.output_block_spec_path, "wb"))
                    pickle.dump(obj=steps, file=open("/storage/yakir/secure_inference/steps.pickle", "wb"))
                    pickle.dump(obj=distorted_losses, file=open("/storage/yakir/secure_inference/distorted_losses.pickle", "wb"))
                    pickle.dump(obj=baseline_losses, file=open("/storage/yakir/secure_inference/baseline_losses.pickle", "wb"))



if __name__ == "__main__":
    # checkpoint = "/home/yakir/epoch_14.pth"
    # input_block_spec_path = "/home/yakir/block_size_spec_4x4_algo.pickle"
    # output_block_spec_path = "/home/yakir/block_size_spec_4x4_algo_out.pickle"
    # config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"

    checkpoint = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
    input_block_spec_path = "./relu_spec_files/classification/resnet50_8xb32_in1k/iterative/num_iters_1/iter_0/block_size_spec_4x4_algo.pickle"
    output_block_spec_path = "./relu_spec_files/classification/resnet50_8xb32_in1k/iterative/num_iters_1/iter_0/block_size_spec_4x4_algo_simulated_annealing_v4.pickle"
    config = "/storage/yakir/secure_inference/research/configs/classification/resnet/iterative/iter01_algo4x4_0.001_4_baseline.py"

    # block_size_spec = pickle.load(open(input_block_spec, 'rb'))
    gpu_id = 0

    cfg = mmcv.Config.fromfile(config)
    gpu_id = gpu_id
    params = param_factory(cfg)

    params.CHECKPOINT = checkpoint
    with torch.no_grad():
        SimulatedAnnealingHandler(gpu_id=gpu_id,
                                  params=params,
                                  cfg=cfg,
                                  input_block_spec_path=input_block_spec_path,
                                  output_block_spec_path=output_block_spec_path).extract_deformation_channel_ord(iteations=100000000)