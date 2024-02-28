FROM python:3.9

ENV TZ Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip3 install xgboost pandas numpy joblib scikit-learn  matplotlib pysnooper

COPY job/xgb/* /app/
COPY job/pkgs /app/job/pkgs
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]



