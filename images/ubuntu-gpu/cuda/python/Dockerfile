
ARG FROM_IMAGES
ARG PYTHON_VERSION
FROM $FROM_IMAGES

# 安装python
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y $PYTHON_VERSION libsasl2-dev libpq-dev python3-pip python3-distutils \
    && rm -rf /usr/bin/python &&  ln -s /usr/bin/$PYTHON_VERSION /usr/bin/python

RUN pip install --upgrade pip && pip install flask requests kubernetes numpy pandas tornado pysnooper


