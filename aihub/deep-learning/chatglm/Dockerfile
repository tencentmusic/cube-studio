# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:chatglm  .
FROM zxdu20/glm-cuda112:latest

WORKDIR /app
USER root
WORKDIR /workspace

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

# 安装中文，和基础的apt工具包
COPY sources.list /etc/apt/sources.list
RUN apt update -y

RUN pip install --upgrade pip

# glm
COPY requirements.txt /workspace/ChatGLM-6B_requirements.txt
RUN pip install -r ChatGLM-6B_requirements.txt -i https://pypi.douban.com/simple
RUN pip install -U deep_training cpm_kernels icetk deepspeed -i https://pypi.douban.com/simple

# 安装运维工具
RUN apt install -y -f --no-install-recommends apt-transport-https gnupg2 jq dnsutils iputils-ping net-tools mysql-client locales zip unzip nginx lsof

# 配置cube-studio基础环境
RUN pip3 install celery redis pyarrow requests_toolbelt cryptography tqdm fsspec aiohttp librosa flask werkzeug requests Pillow pysnooper opencv-python numpy && rm -rf ~/.cache

ENV PYTHONPATH /src/:/github/:$PYTHONPATH

WORKDIR /app

COPY * /app/
COPY init.sh /init.sh
RUN bash /init.sh
