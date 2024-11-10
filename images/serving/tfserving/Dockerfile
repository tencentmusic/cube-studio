
ARG FROM_IMAGES
FROM $FROM_IMAGES
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Shanghai

RUN apt update -y ; apt-get install -y wget

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https software-properties-common gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl locales zip unzip

# 安装中文
RUN apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

# 便捷操作
RUN echo "alias ll='ls -alF'" >> ~/.bashrc && \
    echo "alias la='ls -A'" >> ~/.bashrc && \
    echo "alias vi='vim'" >> ~/.bashrc

