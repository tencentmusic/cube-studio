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


