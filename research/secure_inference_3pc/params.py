class Params:
    # CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
    SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    MODEL_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
    RELU_SPEC_FILE = None #"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/test/block_size_spec_0.15.pickle"
    # SECURE_CONFIG_PATH = "/storage/yakir/secure_inference/research/pipeline/configs/m-v2_256x256_ade20k/baseline_secure.py"
    # MODEL_PATH = "/storage/yakir/secure_inference/work_dirs/m-v2_256x256_ade20k/relu_spec_0.15/iter_160000.pth"
    # RELU_SPEC_FILE = "/storage/yakir/secure_inference/block_size_spec_0.15.pickle"
    IMAGE_SHAPE = (1, 3, 256, 256)
    NUM_IMAGES = 1
    DUMMY_RELU = False
    PRF_PREFETCH = True
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second