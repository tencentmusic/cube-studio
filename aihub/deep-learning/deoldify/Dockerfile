
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9

RUN git clone https://github.com/jantic/DeOldify.git && pip3 install -r DeOldify/requirements.txt && mkdir /DeOldify/models && wget https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -O /DeOldify/models/ColorizeArtistic_gen.pth
ENV PYTHONPATH /DeOldify:/app:$PYTHONPATH
# 安装文件
WORKDIR /app
COPY * /app/

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh



