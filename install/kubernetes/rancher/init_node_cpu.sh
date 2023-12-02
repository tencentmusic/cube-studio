# ubuntu:20.04
#sysctl -w net/netfilter/nf_conntrack_max=524288

service docker stop
rpm -qa | grep docker | xargs yum remove -y
rpm -qa | grep docker
rm -rf /usr/lib/systemd/system/docker.service

# 创建docker的存储路径，因为docker使用的存储很大，默认根目录磁盘很小
rm -rf /data/docker && mkdir -p /data/docker/

# 安装docker，顺带安装调试工具
yum update -y
yum install docker-ce -y 
yum install -y htop docker-compose
yum install -y wireshark
yum install -y telnet

# 停止docker，修改配置
systemctl stop docker
systemctl stop docker.socket
systemctl stop docker.service

# 将源docker目录下文件，复制到新目录下
cp -R /var/lib/docker/* /data/docker/

# 将私有仓库写入到配置文件中
mkdir -p /etc/docker/

(
cat << EOF
{
    "insecure-registries":["docker.oa.com:8080"],
    "data-root": "/data/docker"
}
EOF
)> /etc/docker/daemon.json

systemctl daemon-reload
# 重启docker
systemctl start docker 
# 删除原目录数据
rm -rf /var/lib/docker
