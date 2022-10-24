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

# 配置spark-defaults.conf
echo "spark.ui.enabled=false" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.port=${PORT1}" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.blockManager.port=${PORT2}" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.bindAddress=0.0.0.0" >> ${SPARK_HOME}/conf/spark-defaults.conf
echo "spark.driver.host=${SERVICE_EXTERNAL_IP}" >>${SPARK_HOME}/conf/spark-defaults.conf



