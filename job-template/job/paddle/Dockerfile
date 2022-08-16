FROM ubuntu:18.04
RUN apt-get update && apt-get -y install gcc g++ libjpeg-dev zlib1g-dev cmake

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip
# 安装python
RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

RUN wget https://github.com/stern/stern/releases/download/v1.21.0/stern_1.21.0_linux_amd64.tar.gz && tar -zxvf stern_1.21.0_linux_amd64.tar.gz && rm stern_1.21.0_linux_amd64.tar.gz && chmod +x stern &&  mv stern /usr/bin/stern

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

RUN pip install kubernetes==20.13.0 pysnooper psutil
COPY job/paddle/* /app/
COPY job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]


