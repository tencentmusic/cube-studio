# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base  -f Dockerfile .

FROM ubuntu:20.04

RUN apt update
ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

# 安装运维工具
RUN apt install -y -f --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip unzip nginx lsof

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

# 引入nginx
RUN mkdir -p /data/log/nginx/
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf
COPY entrypoint.sh /entrypoint.sh
ENV PYTHONPATH /src/:/github/:$PYTHONPATH
WORKDIR /app
ENTRYPOINT ["bash", "entrypoint.sh"]

