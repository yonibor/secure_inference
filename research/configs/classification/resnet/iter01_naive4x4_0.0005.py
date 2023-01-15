_base_ = [
    '../_base_/models/resnet50_avg_pool.py', '../_base_/datasets/imagenet_bs128.py',
    '../_base_/schedules/imagenet_bs256_finetune_0.0005.py', '../_base_/default_runtime.py'
]

relu_spec_file = "./relu_spec_files/classification/resnet50_8xb32_in1k/iterative/num_iters_1/iter_0/block_size_spec_4x4_naive.pickle"
load_from = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
work_dir = "./outputs/classification/resnet50_8xb32_in1k/iter01_naive4x4_0.0005"
