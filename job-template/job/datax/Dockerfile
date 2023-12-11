

FROM ubuntu:20.04
ARG TARGETARCH=amd64
ARG PYTHON_VERSION=python3.9
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

COPY job/pkgs/sources.list.${TARGETARCH} /etc/apt/sources.list
COPY job/pkgs/pip.conf /root/.pip/pip.conf
# 安装运维工具
RUN apt update -y && apt install -y --no-install-recommends software-properties-common vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip

# 安装python
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y  libsasl2-dev libpq-dev python3-pip

RUN apt install -y python3.9 python3.9-dev && rm -rf /usr/bin/python3; ln -s /usr/bin/python3.9 /usr/bin/python3 \
    && rm -rf /usr/bin/python;  ln -s /usr/bin/python3 /usr/bin/python

RUN bash -c "wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --ignore-installed --force-reinstall" \
    && rm -rf /usr/bin/pip; ln -s /usr/bin/pip3 /usr/bin/pip

RUN rm -rf /usr/bin/python;  ln -s /usr/bin/python3 /usr/bin/python


# 安装datax
RUN wget https://datax-opensource.oss-cn-hangzhou.aliyuncs.com/202303/datax.tar.gz && tar -zxvf datax.tar.gz -C /usr/local/ && rm -rf datax.tar.gz

# 安装中文
RUN apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

### 安装java
RUN apt install -y openjdk-8-jdk

WORKDIR /usr/local/datax/bin

COPY job/datax/start.sh /usr/local/datax/bin/start.sh

ENTRYPOINT ["bash","start.sh"]
