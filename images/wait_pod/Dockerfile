# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/wait-pod:v1 .
# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/wait-pod:v1 . --push
FROM python:3.9

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install pysnooper psutil requests numpy scikit-learn kubernetes==21.7.0
COPY . /app/

WORKDIR /app
ENV PYTHONPATH=/app:${PYTHONPATH:-}

ENTRYPOINT ["python3", "launcher.py"]


