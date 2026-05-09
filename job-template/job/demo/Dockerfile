FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -y vim wget
RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install pysnooper==1.2.0 requests==2.32.3 numpy==2.0.0 scikit-learn==1.5.1 pandas==2.2.2 pyinstaller==6.8.0 tinyaes==1.1.0 joblib==1.4.2 argparse==1.4.0 matplotlib==3.9.1 redis==5.0.7

COPY . /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python","launcher.py"]


