FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install pysnooper psutil requests huggingface_hub hf_transfer
COPY job/dataset/* /app/

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]


