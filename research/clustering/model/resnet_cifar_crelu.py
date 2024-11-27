import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcls.models.backbones import ResNet_CIFAR
from mmcls.models.backbones.resnet import BasicBlock
from mmcls.models.builder import BACKBONES

from research.clustering.model_handling import get_layer, set_layer
from research.distortion.parameters.classification.resent.resnet18_8xb16_cifar100 import (
    Params,
)

from .crelu_block import ClusterRelu


class BasicBlockCrelu(BasicBlock):

    def __init__(self, **kwargs):
        super(BasicBlockCrelu, self).__init__(**kwargs)

        self.relu_1 = ClusterRelu(is_dummy=True)
        self.relu_2 = ClusterRelu(is_dummy=True)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu_1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu_2(out)

        return out


@BACKBONES.register_module()
class ResNet_CIFAR_CRELU(ResNet_CIFAR):

    arch_settings = {
        18: (BasicBlockCrelu, (2, 2, 2, 2)),
        34: (BasicBlockCrelu, (3, 4, 6, 3)),
        50: (BasicBlockCrelu, (3, 4, 6, 3)),
        101: (BasicBlockCrelu, (3, 4, 23, 3)),
        152: (BasicBlockCrelu, (3, 8, 36, 3)),
    }

    def __init__(self, **kwargs):
        super(ResNet_CIFAR_CRELU, self).__init__(**kwargs)
        assert kwargs["depth"] == 18  # only supported currently

        layers_to_dims = Params().LAYER_NAME_TO_DIMS

        for layer_name, (C, H, W) in layers_to_dims.items():
            if layer_name == "stem":
                layer = ClusterRelu(is_dummy=True, C=C, H=H, W=W)
                set_layer(self, layer_name, layer)
            else:
                layer: ClusterRelu = get_layer(self, layer_name)
            layer.set_default_values(set_size=True, C=C, H=H, W=W)
