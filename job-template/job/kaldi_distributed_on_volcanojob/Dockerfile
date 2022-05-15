FROM ubuntu:18.04

## 换内网源
#COPY job/pkgs/config/ubuntu-sources.list /etc/apt/sources.list
#COPY job/pkgs/config/pip.conf /root/.pip/pip.conf

# 安装运维工具
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq wget git dnsutils iputils-ping net-tools curl locales zip tzdata

# 安装python
RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# 安装stern
RUN wget https://github.com/wercker/stern/releases/download/1.11.0/stern_linux_amd64 && chmod +x stern_linux_amd64 && mv stern_linux_amd64 /usr/bin/stern
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && mv kubectl /usr/bin/

# 安装中文
RUN apt install -y --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

# 设置时区
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# 便捷操作
RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias vi='vim'" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc"

RUN pip install kubernetes==12.0.1 pysnooper psutil
COPY job/kaldi_distributed_on_volcanojob/* /app/
COPY job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]
