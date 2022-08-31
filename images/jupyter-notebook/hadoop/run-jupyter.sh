#!/bin/bash

HOST_IP=$1

# Hadoop生态集群的环境变量统一设置在/opt/third/hadoop-env文件中。

# 设置Hadoop环境变量
echo "export HADOOP_CONF_DIR=/opt/third/hadoop/etc/hadoop" >> /opt/third/hadoop-env

SPARK_HOME="/opt/third/spark"

# 设置Spark环境变量
echo "export SPARK_HOME=${SPARK_HOME}" >> /opt/third/hadoop-env
echo 'export PATH=$PATH:$SPARK_HOME/bin' >> /opt/third/hadoop-env
echo 'export PYTHONPATH=${SPARK_HOME}/python:$(ZIPS=("$SPARK_HOME"/python/lib/*.zip); IFS=:; echo "${ZIPS[*]}"):$PYTHONPATH' >> /opt/third/hadoop-env


# 配置spark-defaults.conf
echo "spark.ui.enabled=false" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.port=32788" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.blockManager.port=32789" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.bindAddress=0.0.0.0" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.host=${HOST_IP}" >>${SPARK_HOME}/conf/spark-defaults.conf


# 设置环境变量到全局/etc/profile
echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> /etc/profile
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
echo 'export M2_HOME=/opt/third/maven' >> /etc/profile
echo 'export PATH=$PATH:$M2_HOME/bin' >> /etc/profile

source /etc/profile
source /opt/third/hadoop-env

# 绑定到/data目录下
jupyter lab --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=3000 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' 2>&1