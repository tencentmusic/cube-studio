
ARG FROM_IMAGES
FROM $FROM_IMAGES

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8
ENV TZ Asia/Shanghai

# 切换内部源
COPY ubuntu-sources.list /etc/apt/sources.list
#RUN wget https://docker-76009.sz.gfp.tencent-cloud.com/tencent/ubuntu-sources.list && cp ubuntu-sources.list /etc/apt/sources.list
RUN apt update -y ; apt-get install -y wget

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https software-properties-common gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl locales zip unzip

# 安装python
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update -y ; apt install -y  libsasl2-dev libpq-dev python-distutils-extra python3-distutils
RUN set -x; rm -rf /usr/bin/python; apt install -y --fix-missing python3.8 && ln -s /usr/bin/python3.8 /usr/bin/python

RUN bash -c "wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --ignore-installed" \
    && rm -rf /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip && pip install pip --upgrade

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

