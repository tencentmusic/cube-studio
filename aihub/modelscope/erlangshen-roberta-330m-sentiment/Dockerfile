
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/modelscope:erlangshen-roberta-330m-sentiment  .
FROM ccr.ccs.tencentyun.com/cube-studio/modelscope:base-cuda11.3-python3.7
# 安装基础环境
WORKDIR /

COPY init.sh /init.sh
COPY * /app/

RUN cd /app/ && python download_model.py

RUN bash /init.sh
# 安装文件
WORKDIR /app
#ENTRYPOINT ["python", "app.py"]
