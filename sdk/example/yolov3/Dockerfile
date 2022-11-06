
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:yolov3  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.6

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh

# 安装文件
WORKDIR /app
COPY * /app/

