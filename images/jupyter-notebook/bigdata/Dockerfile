FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base

MAINTAINER hamawhite


RUN apt-get update && apt install -y lsof

# 下载apache spark安装包
RUN mkdir -p /opt/third && cd /opt/third \
    && wget http://dlcdn.apache.org/spark/spark-3.1.3/spark-3.1.3-bin-hadoop3.2.tgz \
    && tar -xvzf spark-3.1.3-bin-hadoop3.2.tgz \
    && ln -s spark-3.1.3-bin-hadoop3.2 spark \
    && rm -rf spark-3.1.3-bin-hadoop3.2.tgz \
#   创建spark-defaults.conf
    && cd /opt/third/spark/conf \
    && mv spark-defaults.conf.template spark-defaults.conf \
#   安装pyflink
    && pip install apache-flink==1.15.1 \
    && pip install pyspark==3.1.3 \
#   安装JDK8
    && rm -rf /usr/lib/jvm/ \
    && apt-get install -y openjdk-8-jdk \
#   安装maven
    && cd /opt/third \
    && wget http://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz \
    && tar -xvzf apache-maven-3.8.6-bin.tar.gz \
    && ln -s apache-maven-3.8.6 maven \
    && rm -rf apache-maven-3.8.6-bin.tar.gz


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

# 环境初始化配置
COPY init.sh /init.sh