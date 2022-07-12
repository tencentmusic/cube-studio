FROM ubuntu:18.04
COPY job/pkgs/config/ubuntu-sources.list /etc/apt/sources.list
RUN apt-get update && apt-get -y install gcc g++ libjpeg-dev zlib1g-dev cmake

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip
# 安装python
RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

RUN wget https://github.com/stern/stern/releases/download/v1.21.0/stern_1.21.0_linux_amd64.tar.gz && tar -zxvf stern_1.21.0_linux_amd64.tar.gz && rm stern_1.21.0_linux_amd64.tar.gz && chmod +x stern &&  mv stern /usr/bin/stern
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && mv kubectl /usr/bin/
# kubectl logs -f --tail 10 -n pipeline --timestamps --log-flush-frequency=5s --ignore-errors=true --selector tf-job-name=tfjob-rectabdeepfm-gbdt-54bb36ca58cd428d9b0e97d7ba652e

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

RUN pip install kubernetes==12.0.1 pysnooper psutil
COPY job/pytorch_distributed_train_k8s/* /app/
#COPY job/pkgs/config/ubuntu-sources.list /etc/apt/sources.list
COPY job/pkgs/config/pip.conf /root/.pip/pip.conf
COPY job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "pytorchjob_launcher.py"]


