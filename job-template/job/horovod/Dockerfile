FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

RUN wget https://github.com/stern/stern/releases/download/v1.21.0/stern_1.21.0_linux_amd64.tar.gz && tar -zxvf stern_1.21.0_linux_amd64.tar.gz && rm stern_1.21.0_linux_amd64.tar.gz && chmod +x stern &&  mv stern /usr/bin/stern

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install pysnooper psutil requests numpy scikit-learn pandas pyinstaller kubernetes==18.20.0

COPY job/horovod/* /app/
COPY job/pkgs /app/job/pkgs

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]


