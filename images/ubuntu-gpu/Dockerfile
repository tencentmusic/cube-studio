# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/python:3.9 .
# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/python:3.9 . --push

FROM ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install  pysnooper psutil requests numpy  pyinstaller argparse pika Pillow torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 tensorboardX