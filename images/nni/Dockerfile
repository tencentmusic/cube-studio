# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# docker build -t ai.tencentmusic.com/tme-public/nni:20211002 .
# docker run --name nni -it -v $PWD:/app -p 8888:8888 ai.tencentmusic.com/tme-public/nni:20211002 bash
# docker run --name nni -it -p 8888:8888 ai.tencentmusic.com/tme-public/nni:20211002 bash


FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ARG NNI_RELEASE

LABEL maintainer='Microsoft NNI Team<nni@microsoft.com>'

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
    python3.6 \
    python3-dev \
    python3-pip \
    python3-tk \
    libcupti-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*



#
# generate python script
#
RUN ln -s python3 /usr/bin/python

#
# update pip
#
RUN python3 -m pip install --upgrade pip==20.2.4 setuptools==50.3.2

# numpy 1.19.5  scipy 1.5.4
RUN python3 -m pip --no-cache-dir install numpy==1.19.5 scipy==1.5.4

#
# TensorFlow
#
RUN python3 -m pip --no-cache-dir install tensorflow==2.3.1

#
# Keras
#
RUN python3 -m pip --no-cache-dir install Keras==2.4.3

#
# PyTorch
#
RUN python3 -m pip --no-cache-dir install torch==1.7.1 torchvision==0.8.2 pytorch-lightning==1.3.3

#
# sklearn 0.24.1
#
RUN python3 -m pip --no-cache-dir install scikit-learn==0.24.1

#
# pandas==0.23.4 lightgbm==2.2.2
#
RUN python3 -m pip --no-cache-dir install pandas==1.1 lightgbm==2.2.2

RUN pip install jupyter jupyterlab numpy==1.19

## Install NNI
##
#COPY dist/nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl .
#RUN python3 -m pip install nni-${NNI_RELEASE}-py3-none-manylinux1_x86_64.whl

RUN git clone https://github.com/Microsoft/nni.git && cd nni && python3 -m pip install --upgrade pip setuptools && python3 setup.py develop

#RUN apt update && apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
#ENV LANG zh_CN.UTF-8
#ENV LC_ALL zh_CN.UTF-8
#ENV LANGUAGE zh_CN.UTF-8

##
## Vision patch. Need del later
##
COPY interim_vision_patch.py .
RUN python3 interim_vision_patch.py

#
# install aml package
#
RUN python3 -m pip --no-cache-dir install azureml
RUN python3 -m pip --no-cache-dir install azureml-sdk

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbincd

WORKDIR /root



# nnictl create --config examples/trials/mnist-pytorch/config.yml -p 8888 --foreground --url_prefix nni/test
# /app/nni_node/node --max-old-space-size=4096 /app/nni_node/main.js --port 8888 --mode local --experiment_id NjGiK65V --start_mode new --log_dir /root/nni-experiments --log_level info --foreground true --url_prefix nni/test


