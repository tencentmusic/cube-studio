FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu22.04

RUN apt-get update && apt install -y lsof

RUN mkdir -p /examples/ && git clone https://github.com/data-infra/cube-studio.git && cp -r cube-studio/aihub/machine-learning/* /examples/ && rm -rf cube-studio

# 爬虫包
RUN pip install pyquery requests lxml beautifulsoup4 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 数据库查询包
RUN pip install pymysql pika presto-python-client neo4j elasticsearch psycopg2-binary sqlalchemy torndb kafka paramiko prometheus_client oss2 happybase clickhouse-sqlalchemy clickhouse-driver pymongo presto-python-client \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 数据挖掘包
RUN pip install numpy pandas sklearn wheel SciPy pyarrow Pillow PyML MDP Theano opencv-python plotly tqdm gbdt xgboost lightgbm \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 可视化包
RUN pip install matplotlib pyecharts \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 数据服务包
RUN pip install gevent flask asyncio uvloop aiohttp grpcio protobuf grpcio-tools pysnooper \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# 拷贝examples
COPY examples/* /examples/

# 环境初始化配置
COPY init.sh /init.sh