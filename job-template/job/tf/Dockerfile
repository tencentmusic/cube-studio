FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

ARG TARGETARCH=amd64

RUN wget https://github.com/stern/stern/releases/download/v1.21.0/stern_1.21.0_linux_${TARGETARCH}.tar.gz && tar -zxvf stern_1.21.0_linux_${TARGETARCH}.tar.gz && rm stern_1.21.0_linux_${TARGETARCH}.tar.gz && chmod +x stern &&  mv stern /usr/bin/stern

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install kubernetes==21.7.0 pysnooper psutil requests numpy  pyinstaller argparse
COPY job/tf/* /app/
COPY job/pkgs /app/job/pkgs
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]

