FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base

RUN apt-get update && apt install -y lsof

# 修改python3的软链接
RUN cd /usr/bin \
    && rm -rf python3 \
    && ln -s python3.8* python3 \

RUN mkdir -p /examples/ && git clone https://github.com/tencentmusic/cube-studio.git && cp -r cube-studio/aihub/machine-learning/* /examples/ && rm -rf cube-studio

# 爬虫包
RUN pip install pyquery requests lxml beautifulsoup4

# 数据库查询包
RUN pip install pymysql pika presto-python-client neo4j elasticsearch psycopg2-binary sqlalchemy torndb kafka paramiko prometheus_client oss2 happybase clickhouse-sqlalchemy clickhouse-driver pymongo presto-python-client

# 数据挖掘包
RUN pip install numpy pandas sklearn wheel SciPy pyarrow Pillow PyML MDP Theano opencv-python plotly tqdm optimization gbdt xgboost lightgbm

# 可视化包
RUN pip install matplotlib pyecharts

# 数据服务包
RUN pip install gevent flask asyncio uvloop aiohttp grpcio protobuf grpcio-tools pysnooper
# 拷贝examples
COPY examples/* /examples/

# 环境初始化配置
COPY init.sh /init.sh