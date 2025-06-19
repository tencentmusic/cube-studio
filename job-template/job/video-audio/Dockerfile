# 这个下面包含了rayproject/ray:nightly  和  rayproject/ray:nightly-aarch64
FROM ccr.ccs.tencentyun.com/cube-studio/ray:nightly
USER root
ARG TARGETARCH
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
COPY job/pkgs/ubuntu22.04.sources.list.$TARGETARCH /etc/apt/sources.list
# 安装调试相关工具
RUN apt update -y ; apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip software-properties-common

# 安装开发相关工具
RUN apt install -y python3-dev gcc automake autoconf libtool make ffmpeg build-essential

# 安装pip库
RUN pip install kubernetes==25.3.0 pysnooper psutil requests numpy  pyinstaller argparse cython

ENV RAY_CLIENT_SERVER_MAX_THREADS=1000

# 安装当前代码
USER root
COPY job/video-audio/ /app/
COPY job/pkgs /app/job/pkgs

ENV PYTHONPATH=/app:${PYTHONPATH:-}

WORKDIR /app

ENTRYPOINT ["python", "start_download.py"]


