
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh
ENV PYTHONPATH=/DeOldify:$PYTHONPATH
# 安装文件
WORKDIR /app
COPY * /app/


