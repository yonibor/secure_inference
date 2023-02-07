import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
from research.distortion.distortion_utils import get_block_spec_num_relus
spec = "/home/yakir/specs/seg_specs/4x4.pickle"
config = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py"
x = pickle.load(open(spec, 'rb'))
import numpy as np
import mmcv
from research.distortion.parameters.factory import param_factory
plt.subplot(211)

cfg = mmcv.Config.fromfile(config)
params = param_factory(cfg)

reduction = []
for layer_name in x:
    reduction.append(get_block_spec_num_relus({layer_name: x[layer_name]}, params) / get_block_spec_num_relus({layer_name: np.ones_like(x[layer_name])}, params))

plt.bar(range(len(reduction)), reduction, color="#3399e6")

spec = "/home/yakir/specs/cls_specs/4x4.pickle"
config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"


plt.subplot(212)
x = pickle.load(open(spec, 'rb'))
import numpy as np
import mmcv
from research.distortion.parameters.factory import param_factory

cfg = mmcv.Config.fromfile(config)
params = param_factory(cfg)

reduction = []
for layer_name in x:
    reduction.append(get_block_spec_num_relus({layer_name: x[layer_name]}, params) / get_block_spec_num_relus({layer_name: np.ones_like(x[layer_name])}, params))

plt.bar(range(len(reduction)), reduction, color="#3399e6")
plt.tight_layout()
plt.savefig("/home/yakir/Figure_5.png")
