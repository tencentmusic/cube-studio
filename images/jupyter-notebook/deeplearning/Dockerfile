FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base

RUN apt-get update && apt install -y lsof

RUN mkdir -p /examples/ && git clone https://github.com/tencentmusic/cube-studio.git && cp -r cube-studio/aihub/deep-learning/* /examples/ && rm -rf cube-studio

# 数据挖掘包
RUN pip install numpy pandas StatsModels Matplotlib Seaborn  sklearn wheel SciPy pyarrow Pillow PyML MDP Theano opencv-python plotly Bokeh Pydot tqdm gbdt xgboost lightgbm Eli5 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache
# 深度学习包
#
#RUN pip install keras \
#    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 拷贝examples
COPY examples/* /examples/

# 环境初始化配置
COPY init.sh /init.sh