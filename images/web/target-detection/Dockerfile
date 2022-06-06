# docker build -t ai.tencentmusic.com/tme-public/target-detection  .

FROM ubuntu:18.04

RUN apt update

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip

# 安装python
RUN apt install -y python3.6-dev python3-pip libsasl2-dev libpq-dev \
	&& ln -s /usr/bin/python3 /usr/bin/python \
	&& ln -s /usr/bin/pip3 /usr/bin/pip \
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

RUN apt install -y libgl1-mesa-glx
RUN pip install flask werkzeug requests tornado
RUN pip install Pillow pysnooper opencv-python

RUN pip3 install https://github.com/danielgatis/darknetpy/raw/master/dist/darknetpy-4.1-cp36-cp36m-linux_x86_64.whl
WORKDIR /app
#RUN wget https://pengluan-76009.sz.gfp.tencent-cloud.com/github/yolov3.zip && apt install unzip && unzip yolov3.zip && rm yolov3.zip
COPY . /app/
ENTRYPOINT ["python", "server-web.py"]
# docker run --name darknet --privileged -it --rm -v $PWD:/app -p 8080:8080 --entrypoint='' ai.tencentmusic.com/tme-public/target-detection  bash

