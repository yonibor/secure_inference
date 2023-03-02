RELU_SPEC_FILE=/home/ubuntu/specs/cifar_lightweight/0.03.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/classification/resnet/resnet18_cifar100/lightweight.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --secure_config_path $SECURE_CONFIG_PATH --relu_spec_file $RELU_SPEC_FILE  &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --secure_config_path $SECURE_CONFIG_PATH  &

