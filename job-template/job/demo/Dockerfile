FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y && apt install -y vim wget
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install pysnooper requests numpy scikit-learn pandas pyinstaller tinyaes joblib argparse matplotlib redis

COPY * /app/

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python","launcher.py"]


