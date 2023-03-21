# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:chatglm  .
FROM zxdu20/glm-cuda112:latest

WORKDIR /app
USER root
WORKDIR /workspace
# 安装中文，和基础的apt工具包
COPY sources.list /etc/apt/sources.list
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone
RUN apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
RUN apt install -y --force-yes --no-install-recommends -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true software-properties-common vim apt-transport-https gnupg2 ca-certificates-java jq wget git dnsutils iputils-ping net-tools curl lrzsz locales python3-distutils

RUN locale-gen zh_CN && locale-gen zh_CN.utf8

ENV LANG zh_CN.UTF-8
ENV LC_ALL zh_CN.UTF-8
ENV LANGUAGE zh_CN.UTF-8

RUN echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias la='ls -A'" >> /root/.bashrc && \
    echo "alias vi='vim'" >> /root/.bashrc && \
    /bin/bash -c "source /root/.bashrc"

RUN pip install --upgrade pip

# glm
COPY requirements.txt /workspace/ChatGLM-6B_requirements.txt
RUN pip install -r ChatGLM-6B_requirements.txt -i https://pypi.douban.com/simple
RUN pip install -U deep_training cpm_kernels icetk deepspeed -i https://pypi.douban.com/simple

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

# 安装运维工具
RUN apt install -y -f --no-install-recommends apt-transport-https gnupg2 jq dnsutils iputils-ping net-tools mysql-client locales zip unzip nginx lsof

# 配置cube-studio基础环境
RUN pip3 install celery redis pyarrow requests_toolbelt cryptography tqdm fsspec aiohttp librosa flask werkzeug requests Pillow pysnooper opencv-python numpy && rm -rf ~/.cache

ENV PYTHONPATH /src/:/github/:$PYTHONPATH

WORKDIR /app