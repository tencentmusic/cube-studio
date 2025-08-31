source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

pip3 install pyyaml
pip3 install setuptools

pip3 install torch==2.1.0 torchvision==0.16.0 tensorboardX pysnooper requests
pip3 install torch-npu==2.1.0.post8


mkdir -p  data/MNIST/raw/

python demo.py

