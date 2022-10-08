# docker build -t ccr.ccs.tencentyun.com/cube-studio/ai-photo  .

FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

RUN apt update
ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

# 安装运维工具
RUN apt install -y -f --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip unzip

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

# 安装anaconda
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH "/root/miniconda3/bin":$PATH
RUN conda config --set auto_activate_base true

# 安装基础依赖
RUN apt install -y -f python3-opencv
RUN pip install flask werkzeug requests tornado Pillow pysnooper opencv-python numpy scipy matplotlib scikit-image scikit-learn ipython pyyaml visualdl tqdm filelock scipy prettytable torch torchvision paddlepaddle paddleseg
#RUN pip install --upgrade setuptools
RUN conda install -y -c conda-forge dlib

WORKDIR /app
COPY . /app/

# 下载 模型
RUN python download_hubman_seg_inference_models.py
RUN python download_animegan2_inference_models.py
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2 && rm shape_predictor_68_face_landmarks.dat.bz2

ENTRYPOINT ["python", "server-web.py"]
# docker run --name photo --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/ai-photo  bash

