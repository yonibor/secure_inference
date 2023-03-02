RELU_SPEC_FILE=/home/ubuntu/specs/cifar/0.03.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH --dummy_image --image_start 0 --image_end 3 &

