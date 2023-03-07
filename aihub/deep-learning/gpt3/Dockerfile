
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:gpt3  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
COPY * /app/
RUN bash /init.sh
# 安装文件
WORKDIR /app
#ENTRYPOINT ["python", "app.py"]

