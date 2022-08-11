FROM ubuntu:20.04

RUN apt-get update ; apt-get -y --fix-missing install gcc g++ libjpeg-dev zlib1g-dev cmake

# 安装运维工具
RUN apt install -y --no-install-recommends --fix-missing software-properties-common vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl locales zip unzip

RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y  libsasl2-dev libpq-dev python3-pip python3-distutils
RUN set -x; rm -rf /usr/bin/python; apt install -y --fix-missing python3.8 python3.8-dev && ln -s /usr/bin/python3.8 /usr/bin/python

RUN bash -c "wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --ignore-installed" \
    && rm -rf /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

# 安装中文
RUN apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

# 便捷操作
RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias vi='vim'" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc"

RUN pip install kubernetes pysnooper psutil pyspark
# 这里安装spark基础命令，hadoop基础命令，sqoop基础命令，hbase基础命令，对接公司的大数据平台

COPY job/hadoop/* /app/
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["bash", "start.sh"]


