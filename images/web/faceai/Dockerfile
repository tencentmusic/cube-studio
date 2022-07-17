# docker build -t ccr.ccs.tencentyun.com/cube-studio/face:20220701 .

FROM ubuntu:18.04
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

# 安装python
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y  libsasl2-dev libpq-dev python3-pip python3-distutils python3.8-dev cmake libgl1-mesa-glx
RUN set -x; rm -rf /usr/bin/python; apt install -y --fix-missing python3.8 && ln -s /usr/bin/python3.8 /usr/bin/python

RUN bash -c "wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --ignore-installed" \
    && rm -rf /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip && pip install flask requests kubernetes numpy pandas tornado pysnooper gunicorn face_recognition flask_cors opencv-python tensorflow keras

WORKDIR /app
RUN wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/service/face_model.zip && unzip face_model.zip && rm face_model.zip
COPY . /app/

ENTRYPOINT ["bash","-c","gunicorn --bind  0.0.0.0:8080 --workers 5 --timeout 300 --limit-request-line 0 --limit-request-field_size 0 --log-level=info server:app"]

# docker run --rm --name face -it -p 8080:8080 -v $PWD:/app csighub.tencentyun.com/tme-kubeflow/ai:face bash
# gunicorn --bind  0.0.0.0:8080 --workers 20 --timeout 300 --limit-request-line 0 --limit-request-field_size 0 --log-level=info server:app
# docker run --rm --name face -it -p 8080:8080 ccr.ccs.tencentyun.com/cube-studio/face:20220701 bash



