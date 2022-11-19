
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:yolov5  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh

# 安装文件
WORKDIR /app
COPY * /app/
COPY /app/upsampling.py /root/miniconda3/envs/py39/lib/python3.9/site-packages/torch/nn/modules/upsampling.py
