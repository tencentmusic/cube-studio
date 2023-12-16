
# docker build -t ccr.ccs.tencentyun.com/cube-studio/nni:20230601 .
# docker run --name nni -it -v $PWD:/app -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20230601 bash
# docker run --name nni -it -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20230601 bash

# 自带了python3.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG TARGETARCH=amd64

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y install \
    sudo \
    apt-utils \
    git \
    curl \
    vim \
    unzip \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    automake \
    openssh-client \
    openssh-server \
    lsof \
    python3-pip \
    python3-tk \
    libcupti-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s python3 /usr/bin/python
# update pip
#
RUN pip install --upgrade pip setuptools

# 安装启动sshd
RUN apt install -y openssh-server openssh-client

# numpy 1.19.5  scipy 1.5.4
RUN pip3 --no-cache-dir install numpy pandas scipy tensorflow Keras torch torchvision pytorch-lightning scikit-learn pandas lightgbm && rm -rf ~/.cache

RUN apt update && apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8
#
# install aml package

RUN python3 -m pip --no-cache-dir install dill pysnooper
RUN pip install  azureml azureml-sdk nni==2.10.1

# RUN git clone -b v2.10.1 https://github.com/microsoft/nni.git && pip install --upgrade setuptools pip wheel && cd nni &&


ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbincd

WORKDIR /root



# nnictl create --config examples/trials/mnist-pytorch/config.yml -p 8888 --foreground --url_prefix nni/test
# /app/nni_node/node --max-old-space-size=4096 /app/nni_node/main.js --port 8888 --mode local --experiment_id NjGiK65V --start_mode new --log_dir /root/nni-experiments --log_level info --foreground true --url_prefix nni/test


