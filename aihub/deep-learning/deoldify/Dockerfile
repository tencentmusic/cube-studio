
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9
RUN pip install numpy
ENV PYTHONPATH /DeOldify:/app:$PYTHONPATH
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh
# 安装文件
WORKDIR /app
COPY * /app/


