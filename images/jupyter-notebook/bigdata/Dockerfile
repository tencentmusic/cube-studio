FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu22.04

MAINTAINER hamawhite
ARG SPARK_HADOOP_VERSION=3.2
ARG SPARK_VERSION=3.1.3
ARG FLINK_VERSION=1.15.1
ARG MAVEN_VERSION=3.8.6
ARG APACHE_MIRROR=http://dlcdn.apache.org
ARG JDK_VERSION=8
ARG PYPI_MIRROR=mirrors.aliyun.com
ARG PIPI_MIRROR_ENABLE=false
ARG UBUNTU_MIRROR=mirrors.ustc.edu.cn
ARG UBUNTU_MIRROR_ENABLE=false

RUN if [ "${UBUNTU_MIRROR_ENABLE}" = "true" ]; then \
        cp /etc/apt/sources.list /etc/apt/sources.list.ori && \
        sed -i s@/archive.ubuntu.com/@/${UBUNTU_MIRROR}/@g /etc/apt/sources.list && \
        sed -i s@/security.ubuntu.com/@/${UBUNTU_MIRROR}/@g /etc/apt/sources.list && \
        apt clean && apt -y update; \
    fi

RUN apt install -y lsof

# pip 镜像
RUN mkdir ~/.pip
RUN if [ "${PIPI_MIRROR_ENABLE}" = "true" ]; then \
        echo "use pip mirror";\
        pip config set global.index-url https://${PYPI_MIRROR}/pypi/simple/; \
        pip config set install.trusted-host ${PYPI_MIRROR}; \
    else \
        echo "use default pip mirror";\
    fi

# 下载apache spark安装包
RUN mkdir -p /opt/third && cd /opt/third \
    && wget ${APACHE_MIRROR}/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${SPARK_HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${SPARK_HADOOP_VERSION}.tgz \
    && ln -s spark-${SPARK_VERSION}-bin-hadoop${SPARK_HADOOP_VERSION} spark \
    && rm -rf spark-${SPARK_VERSION}-bin-hadoop${SPARK_HADOOP_VERSION}.tgz \
#   创建spark-defaults.conf
    && cd /opt/third/spark/conf \
    && mv spark-defaults.conf.template spark-defaults.conf \
#   安装pyflink
    && pip install apache-flink==${FLINK_VERSION}\
#   安装JDK8
    && rm -rf /usr/lib/jvm/ \
    && apt-get install -y openjdk-${JDK_VERSION}-jdk \
#   安装maven
    && cd /opt/third \
    && wget ${APACHE_MIRROR}/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz\
    && tar -xvzf apache-maven-${MAVEN_VERSION}-bin.tar.gz \
    && ln -s apache-maven-${MAVEN_VERSION} maven \
    && rm -rf apache-maven-${MAVEN_VERSION}-bin.tar.gz

# 安装大数据python包
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

# 拷贝examples
COPY examples/* /examples/

# 拷贝示例的hadoop和hive配置文件
COPY conf/hive/* /opt/third/hive/conf/
COPY conf/hadoop/* /opt/third/hadoop/etc/hadoop/

# 新增flink-dep.xml
COPY conf/flink/flink-dep.xml /opt/third/flink/

# 修改maven镜像
COPY maven/conf/settings.xml /opt/third/maven/conf/settings.xml

ENV M2_HOME /opt/third/maven
ENV PATH $M2_HOME/bin:$PATH

# 下载pyflink hivecatalog的依赖
RUN cd /opt/third/flink \
    && mkdir lib \
    && mvn -f flink-dep.xml dependency:copy-dependencies -DoutputDirectory=lib


# 设置环境变量到全局/etc/profile
ENV HADOOP_CONF_DIR /opt/third/hadoop/etc/hadoop
ENV SPARK_HOME /opt/third/spark
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV M2_HOME /opt/third/maven
ENV PATH $PATH:$JAVA_HOME/bin:$M2_HOME/bin:$SPARK_HOME/bin
ENV PYTHONPATH ${SPARK_HOME}/python:${SPARK_HOME}/lib/py4j-0.10.9-src.zip:${SPARK_HOME}/lib/pyspark.zip:${SPARK_HOME}/lib/pyspark.zip:$PYTHONPATH

# 环境初始化配置
COPY init.sh /init.sh



