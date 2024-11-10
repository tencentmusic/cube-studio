
# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/nni:20240501  --build-arg TARGETARCH=amd64 .
# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/nni:20230601 . --push
# docker run --name nni -it -v $PWD:/app -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20230601 bash
# docker run --name nni -it -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20230601 bash

ARG TARGETARCH

FROM ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-${TARGETARCH}

ENV DEBIAN_FRONTEND=noninteractive
# 安装启动sshd
RUN apt-get -y update
RUN apt install -y openssh-server openssh-client libopenblas-dev python3-tk libcupti-dev
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash
RUN apt-get install -y nodejs && npm install yarn -g
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install dill pysnooper azureml azureml-sdk

RUN pip install nni
# arm64版本要用源码安装
# RUN git clone https://github.com/microsoft/nni.git && pip install --upgrade setuptools pip wheel && cd nni &&

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbincd

WORKDIR /root


# nnictl create --config examples/trials/mnist-pytorch/config.yml -p 8888 --foreground --url_prefix nni/test
# /app/nni_node/node --max-old-space-size=4096 /app/nni_node/main.js --port 8888 --mode local --experiment_id NjGiK65V --start_mode new --log_dir /root/nni-experiments --log_level info --foreground true --url_prefix nni/test


