# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base  -f Dockerfile .

#FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
FROM ubuntu:20.04

RUN apt update
ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

# 安装运维工具
RUN apt install -y -f --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip unzip

# 安装中文
RUN apt install -y -f --no-install-recommends fontconfig locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

# 便捷操作
RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
	echo "alias la='ls -A'" >> /root/.bashrc && \
	echo "alias vi='vim'" >> /root/.bashrc && \
	/bin/bash -c "source /root/.bashrc"

# 安装anaconda
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH "/root/miniconda3/bin":$PATH
RUN conda config --set auto_activate_base true

# 安装
RUN apt install -y -f python3-opencv ffmpeg
RUN pip3 install flask werkzeug requests tornado Pillow pysnooper opencv-python numpy scipy matplotlib scikit-image ipython pyyaml visualdl filelock \
    && rm -rf ~/.cache

COPY cubestudio-2022.10.1-py3-none-any.whl /cubestudio-2022.10.1-py3-none-any.whl
RUN pip install /cubestudio-py3-none-any.whl

WORKDIR /app
#COPY entrypoint.sh /entrypoint.sh

#ENTRYPOINT ["bash", "entrypoint.sh"]

# docker run --name aihub --privileged -it -v $PWD:/app --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:base  bash

