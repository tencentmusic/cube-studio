# docker build -t ccr.ccs.tencentyun.com/cube-studio/wenet-mini:latest  .

FROM wenetorg/wenet-mini:latest
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y ; apt install -y wget git

# 安装运维工具
RUN apt install -y --force-yes --no-install-recommends software-properties-common vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y  libsasl2-dev libpq-dev python3-pip python3-distutils

# 安装python
RUN rm -rf /usr/bin/python; ln -s /usr/bin/python3 /usr/bin/python
RUN rm /usr/bin/pip ; ln -s /usr/bin/pip3 /usr/bin/pip && pip install --upgrade pip

# 下载预训练模型
RUN mkdir -p /home/github && cd /home/github && git clone https://github.com/wenet-e2e/wenet.git
RUN mkdir -p /home/wenet && cd /home/wenet && wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_u2pp_conformer_libtorch.tar.gz && tar -xf 20210618_u2pp_conformer_libtorch.tar.gz && mv 20210618_u2pp_conformer_libtorch model  && rm 20210618_u2pp_conformer_libtorch.tar.gz

WORKDIR /home/github/wenet
RUN pip install -r requirements.txt
RUN pip install flask werkzeug requests tornado pysnooper
#RUN pip install pytorch torchvision torchaudio
WORKDIR /home/github/wenet/runtime/server/x86/web
ENV LD_LIBRARY_PATH=/home/wenet/lib
ENV GLOG_logtostderr=1
ENV GLOG_v=2
ENV MODEL=/home/wenet/model

ENTRYPOINT ["bash", "-c","(nohup python app.py --port 80 &) && /home/wenet/websocket_server_main   --port 10086   --chunk_size 16   --model_path $MODEL/final.zip   --unit_path $MODEL/units.txt 2>&1 | tee server.log"]

# docker run --name wenet --privileged -it --rm -v $PWD:/app -p 8080:8080 -p 10086:10086 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/wenet-mini:latest  bash

