# FROM ubuntu:18.04
FROM horovod/horovod:nightly


USER root
RUN apt update && apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip
RUN apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && locale-gen zh_CN && locale-gen zh_CN.utf8
ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8
# 便捷操作
RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias vi='vim'" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc"

RUN pip install pysnooper dill requests kubernetes==18.20.0
ENV TZ 'Asia/Shanghai'

COPY job/horovod/* /app/
COPY job/pkgs /app/job/pkgs
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app

ENTRYPOINT ["python", "start.py"]

# https://github.com/horovod/horovod/tree/master/examples   示例代码
# docker run --name horovod -it --entrypoint '' ccr.ccs.tencentyun.com/cube-studio/horovod:20210401 bash