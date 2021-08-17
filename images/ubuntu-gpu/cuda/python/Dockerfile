
ARG FROM_IMAGES
ARG PYTHON_VERSION
FROM $FROM_IMAGES

# 安装python
RUN add-apt-repository ppa:deadsnakes/ppa && apt install -y $PYTHON_VERSION python3-pip libsasl2-dev libpq-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install flask requests kubernetes numpy pandas tornado pysnooper

