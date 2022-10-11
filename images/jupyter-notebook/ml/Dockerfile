FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base

RUN apt-get update && apt install -y lsof

# 修改python3的软链接
RUN cd /usr/bin \
    && rm -rf python3 \
    && ln -s python3.8* python3
RUN mkdir -p /examples/ && git clone https://github.com/tencentmusic/cube-studio.git && cp -r cube-studio/aihub/machine-learning/* /examples/ && rm -rf cube-studio

# 拷贝examples
COPY examples/* /examples/

# 环境初始化配置
COPY init.sh /init.sh