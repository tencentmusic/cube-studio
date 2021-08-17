hostname=`ifconfig eth1 | grep 'inet '| awk '{print $2}' | head -n 1 | awk -F. {'printf("node%03d%03d%03d%03d\n", $1, $2, $3, $4)'}`
echo $hostname
hostnamectl set-hostname ${hostname}

echo "127.0.0.1 ${hostname}" >> /etc/hosts
echo "::1 ${hostname}" >> /etc/hosts

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
# 停止docker，修改配置
systemctl stop docker
# 将源docker目录下文件，复制到新目录下
cp -R /var/lib/docker/* /data/docker/
# 将docker新目录写入到配置文件中
sed -i "s/containerd.sock/containerd.sock --graph \/data\/docker /g" /usr/lib/systemd/system/docker.service 

# 将私有仓库写入到配置文件中
mkdir -p /etc/docker/
echo '{ "insecure-registries":["docker.oa.com"] } ' > /etc/docker/daemon.json
# 重载配置
systemctl daemon-reload
# 重启docker
systemctl start docker 
# 删除原目录数据
rm -rf /var/lib/docker
