#!/bin/bash

# 配置ssh链接
echo "Port ${SSH_PORT}" >> /etc/ssh/sshd_config
sed -i "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/g" /etc/ssh/sshd_config
sed -i "s/#PermitRootLogin yes/PermitRootLogin yes/g" /etc/ssh/sshd_config
sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/g" /etc/ssh/sshd_config
echo root:cube-studio | chpasswd
service ssh restart
# 客户端连接命令，    ssh -p ${SSH_PORT} root@${SERVICE_EXTERNAL_IP}

sed -i "s/localhost/${SERVICE_EXTERNAL_IP}/g" /examples/ssh连接
sed -i "s/localport/${SSH_PORT}/g" /examples/ssh连接

# 配置example
ln -s /examples /mnt/${USERNAME}/

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
echo "spark.driver.port=${PORT1}" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.blockManager.port=${PORT2}" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.bindAddress=0.0.0.0" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.host=${SERVICE_EXTERNAL_IP}" >>${SPARK_HOME}/conf/spark-defaults.conf


# 设置环境变量到全局/etc/profile
echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> /etc/profile
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
echo 'export M2_HOME=/opt/third/maven' >> /etc/profile
echo 'export PATH=$PATH:$M2_HOME/bin' >> /etc/profile
echo 'source /opt/third/hadoop-env' >> /etc/profile

source /etc/profile


