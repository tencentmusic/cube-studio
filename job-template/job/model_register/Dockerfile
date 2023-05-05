
FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

RUN pip install kubernetes==18.20.0 pysnooper psutil requests

COPY job/model_register/* /app/

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]


