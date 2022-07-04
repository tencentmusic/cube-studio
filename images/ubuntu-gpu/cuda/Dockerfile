ARG FROM_IMAGES
FROM $FROM_IMAGES

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 安装运维工具
RUN apt update; apt install -y --force-yes --no-install-recommends software-properties-common vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip unzip

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


