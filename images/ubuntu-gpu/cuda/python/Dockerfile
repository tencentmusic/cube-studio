ARG FROM_IMAGES
FROM ${FROM_IMAGES}

ARG PYTHON_VERSION

# 安装python
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y  libsasl2-dev libpq-dev python3-pip $PYTHON_VERSION-distutils
RUN set -x; rm -rf /usr/bin/python /usr/bin/python3; apt install -y --fix-missing ${PYTHON_VERSION} && ln -s /usr/bin/${PYTHON_VERSION} /usr/bin/python && ln -s /usr/bin/${PYTHON_VERSION} /usr/bin/python3

RUN bash -c "wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --ignore-installed" \
    && rm -rf /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install -U pip && pip install flask requests kubernetes numpy pandas tornado pysnooper


