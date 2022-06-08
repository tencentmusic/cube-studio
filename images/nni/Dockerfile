
# docker build -t ccr.ccs.tencentyun.com/cube-studio/nni:20211003 .
# docker run --name nni -it -v $PWD:/app -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20211003 bash
# docker run --name nni -it -p 8888:8888 ccr.ccs.tencentyun.com/cube-studio/nni:20211003 bash

#docker run --name nni -it -p 8888:8888 nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 bash
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

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
    libcupti-dev && apt-get clean && rm -rf /var/lib/apt/lists/*


# generate python script
#
RUN ln -s python3 /usr/bin/python

#
# update pip
#
RUN pip3 install --upgrade pip==20.2.4 setuptools==50.3.2

# numpy 1.19.5  scipy 1.5.4
RUN pip3 --no-cache-dir install numpy==1.19.5 scipy==1.5.4 tensorflow==2.3.1 Keras==2.4.3 torch==1.7.1 torchvision==0.8.2 pytorch-lightning==1.3.3 scikit-learn==0.24.1 pandas==1.1 lightgbm==2.2.2 && rm ~/.cache

RUN pip3 install jupyter jupyterlab numpy==1.19

RUN git clone https://github.com/Microsoft/nni.git && cd nni && python3 -m pip install --upgrade pip setuptools && python3 setup.py develop

RUN apt update && apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

RUN cd nni && python3 interim_vision_patch.py

#
# install aml package
#
RUN python3 -m pip --no-cache-dir install azureml azureml-sdk

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/root/.local/bin:/usr/bin:/bin:/sbincd

WORKDIR /root



# nnictl create --config examples/trials/mnist-pytorch/config.yml -p 8888 --foreground --url_prefix nni/test
# /app/nni_node/node --max-old-space-size=4096 /app/nni_node/main.js --port 8888 --mode local --experiment_id NjGiK65V --start_mode new --log_dir /root/nni-experiments --log_level info --foreground true --url_prefix nni/test


