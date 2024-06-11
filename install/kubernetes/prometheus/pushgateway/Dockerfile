# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/prometheus:pushgateway .
FROM ubuntu:18.04
RUN apt update && \
    apt install -y python3.6-dev python3-pip curl iputils-ping net-tools python3-setuptools && \
    apt install -y --force-yes --no-install-recommends locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && \
    locale-gen zh_CN && locale-gen zh_CN.utf8 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    pip install simplejson requests uvloop asyncio paramiko prometheus_client aiohttp webhooks opsdroid aiohttp_cors && \
    rm -rf /root/.cache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/man  /usr/share/doc /usr/share/doc-base

ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

COPY util /app/pushgateway/util
COPY test /app/pushgateway/test
COPY server.py /app/pushgateway/

EXPOSE 80

USER root
WORKDIR /app/pushgateway
CMD ["python","server.py"]





