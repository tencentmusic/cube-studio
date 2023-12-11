#FROM rayproject/ray:nightly
#FROM ccr.ccs.tencentyun.com/cube-studio/ray:gpu
ARG BASE_IMAGE=rayproject/ray:nightly-py39-cpu
FROM ${BASE_IMAGE}
USER root

# 安装调试相关工具
RUN apt update -y && apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip software-properties-common

ENV TZ 'Asia/Shanghai'
ENV DEBIAN_FRONTEND=noninteractive

# 安装开发相关工具
RUN apt install -y python3-dev gcc automake autoconf libtool make ffmpeg build-essential

# 安装pip库
RUN pip install pysnooper cython

# 便捷操作
RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias vi='vim'" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc"

WORKDIR /app
RUN pip3 install kubernetes==21.7.0 pysnooper psutil
RUN pip3 install scikit-learn pandas numpy joblib

COPY job/ray_sklearn/* /app/
COPY job/pkgs /app/job/pkgs
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]
