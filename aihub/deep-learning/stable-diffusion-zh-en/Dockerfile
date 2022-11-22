# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion-zh-en  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.8
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
COPY load.py /load.py
RUN bash /init.sh
# 安装文件
WORKDIR /app
COPY * /app/
COPY /app/predictor.py /root/miniconda3/envs/py38/lib/python3.8/site-packages/flagai/model/predictor/predictor.py /home/cube-studio/

