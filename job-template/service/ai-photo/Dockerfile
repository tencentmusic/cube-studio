# docker build -t ccr.ccs.tencentyun.com/cube-studio/ai-photo  .

FROM ubuntu:18.04

RUN apt update
ENV TZ Asia/Shanghai
# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip unzip

# 安装python
RUN apt install -y python3.8-dev python3.8-pip libsasl2-dev libpq-dev \
	&& rm -rf /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python \
	&& rm -rf /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip \
	&& pip install --upgrade pip


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

RUN apt-get install -y libgl1-mesa-glx build-essential cmake libgtk-3-dev libboost-all-dev pkg-config libx11-dev libatlas-base-dev libboost-python-dev

RUN pip install flask werkzeug requests tornado
RUN pip install Pillow pysnooper opencv-python
RUN pip install numpy scipy matplotlib scikit-image scikit-learn ipython
RUN pip install --upgrade setuptools
RUN pip install pyyaml visualdl opencv-python tqdm filelock scipy prettytable sklearn
#torch torchvision dlib

WORKDIR /app
COPY . /app/
ENTRYPOINT ["python", "server-web.py"]
# docker run --name photo --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/ai-photo  bash

