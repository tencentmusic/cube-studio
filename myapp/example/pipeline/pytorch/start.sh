pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# pip install tensorboardX torch torchvision --no-cache-dir
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 tensorboardX --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 tensorboardX

pip install tensorboardX pysnooper requests

#export NCCL_IB_HCA=mlx5   # 需要适配
#export NCCL_IB_TC=136
# export NCCL_IB_SL=5   # 需要适配，可不填
# export NCCL_IB_GID_INDEX=0    # 需要适配，可不填
#export NCCL_IB_TIMEOUT=22
# export NCCL_SOCKET_IFNAME=eth0  # 可不填，默认就是这个，有些协议无法走ib，会自动走以太网
#export NCCL_DEBUG=INFO

mkdir -p  data/MNIST/raw/
if [ "$RANK" = "0" ]; then

  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/train-images-idx3-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/train-labels-idx1-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/t10k-images-idx3-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/t10k-labels-idx1-ubyte.gz
fi
python demo.py

