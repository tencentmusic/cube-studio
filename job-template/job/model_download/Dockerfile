
FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

RUN pip install kubernetes==21.7.0 pysnooper psutil requests
COPY job/model_download/* /app/

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]


