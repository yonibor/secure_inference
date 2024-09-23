from typing import Tuple
from torch import nn

def _get_layer_details(model, layer_name) -> Tuple[nn.Module, str]:
    if layer_name == "stem":
        block = model.backbone
        name = 'relu'
    else:
        res_layer_name, block_name, relu_name = layer_name.split("_")
        layer = getattr(model.backbone, res_layer_name)
        block = layer._modules[block_name]
        name = f"relu_{relu_name}"
    return block, name

def set_layer(model, layer_name, block_relu: nn.Module):
    block, name = _get_layer_details(model, layer_name)
    setattr(block, name, block_relu)

def get_layer(model, layer_name):
    block, name = _get_layer_details(model, layer_name)
    return getattr(block, name)