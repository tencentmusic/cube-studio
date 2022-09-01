FROM ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base

MAINTAINER hamawhite

COPY hadoop/run-jupyter.sh /root/run-jupyter.sh
# 拷贝examples
COPY hadoop/examples/* /examples/

# 拷贝示例的hadoop和hive配置文件
COPY hadoop/conf/hive/* /opt/third/hive/conf
COPY hadoop/conf/hadoop/* /opt/third/hadoop/etc/hadoop/

# 修改maven镜像
COPY hadoop/maven/conf/settings.xml /opt/third/maven/conf/settings.xml

# 新增flink-dep.xml
COPY hadoop/conf/flink/flink-dep.xml /opt/third/flink/

RUN apt-get update && apt install -y lsof

# 修改python3的软链接
RUN cd /usr/bin \
    && rm -rf python3 \
    && ln -s python3.8* python3

# 安装maven
RUN cd /opt/third \
    && wget http://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz \
    && tar -xvzf apache-maven-3.8.6-bin.tar.gz \
    && ln -s apache-maven-3.8.6 maven \
    && rm -rf apache-maven-3.8.6-bin.tar.gz

ENV M2_HOME /opt/third/maven
ENV PATH $M2_HOME/bin:$PATH

RUN mvn -version
RUN echo "maven success"

# 下载pyflink hivecatalog的依赖
RUN cd /opt/third/flink \
    && mkdir lib \
    && mvn -f flink-dep.xml dependency:copy-dependencies -DoutputDirectory=lib

# 下载apache spark安装包
RUN cd /opt/third \
    && wget http://dlcdn.apache.org/spark/spark-3.1.3/spark-3.1.3-bin-hadoop3.2.tgz \
    && tar -xvzf spark-3.1.3-bin-hadoop3.2.tgz \
    && ln -s spark-3.1.3-bin-hadoop3.2 spark \
    && rm -rf spark-3.1.3-bin-hadoop3.2.tgz \
#   创建spark-defaults.conf
    && cd /opt/third/spark/conf \
    && mv spark-defaults.conf.template spark-defaults.conf \
#   安装pyflink
    && pip install apache-flink==1.15.1 \
#   安装JDK8
    && rm -rf /usr/lib/jvm/ \
    && apt-get install -y openjdk-8-jdk


# 设置环境变量到全局/etc/profile
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> /etc/profile \
    && echo "export PATH=$PATH:$JAVA_HOME/bin" >> /etc/profile \
    && echo "export M2_HOME=/opt/third/maven" >> /etc/profile \
    && echo "export PATH=$PATH:$M2_HOME/bin" >> /etc/profile \
    && /bin/bash -c "source /etc/profile"

ENTRYPOINT ["bash","/root/run-jupyter.sh"]