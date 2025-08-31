stage=${STAGE:-}
# 先检查下面的命令中的参数，比如内网仓库的地址，docker根目录的地址，nfs的地址，rancher server的加入地址，机器的网卡名称
# =================安装docker==================
if [ "$stage" = "1" ]; then

sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release vim git wget net-tools

sudo mkdir -p /etc/apt/keyrings
rm -rf /etc/apt/keyrings/docker.gpg
rm -rf /etc/apt/sources.list.d/docker.list

curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
arch=amd64    # 或者arm64
sudo add-apt-repository  -y "deb [arch=${arch}] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

apt install -y docker-ce=5:27.0.3-1~ubuntu.22.04~jammy

sudo mkdir -p /etc/docker

cat > /etc/docker/daemon.json <<EOF
{
    "registry-mirrors": ["https://hub.rat.dev/","https://docker.xuanyuan.me", "https://docker.m.daocloud.io","https://dockerproxy.com"],
    "dns": ["114.114.114.114","8.8.8.8"],
    "max-concurrent-downloads": 30,
    "data-root": "/data/docker",
    "insecure-registries":["docker.oa.com:8080"]
}
EOF

systemctl stop docker
systemctl daemon-reload
systemctl start docker

fi

# =================检测：安装docker==================
if [ "$stage" = "11" ]; then
  docker ps
fi
## =============挂载nfs================

if [ "$stage" = "2" ]; then
  apt update
  apt install -y nfs-kernel-server
  apt install -y nfs-common
  export server=10.0.0.76
  mkdir -p /data/nfs
  echo "${server}:/data/nfs  /data/nfs   nfs   defaults  0  0" >> /etc/fstab
  mount -a
  mkdir -p /data/nfs/k8s
  ln -s /data/nfs/k8s /data/

fi
# =================检测：挂载nfs==================
if [ "$stage" = "22" ]; then
  df -h |grep nfs
fi
## =================拉取rancher镜像=====================
if [ "$stage" = "3" ]; then
  sh /data/nfs/cube-studio-enterprise/install/kubernetes/rancher/pull_rancher_images.sh
fi
# =================检测：拉取rancher镜像==================
if [ "$stage" = "33" ]; then
  docker images |grep rancher |wc -l
fi

## ==================检测网卡对应的ip是不是对的=====================
if [ "$stage" = "4" ]; then
  ip=`ifconfig eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
  echo $ip
fi
## ==================加入rancher集群=====================
if [ "$stage" = "44" ]; then
  ip=`ifconfig eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
  sudo docker run -d --privileged --restart=unless-stopped --net=host -v /etc/kubernetes:/etc/kubernetes -v /var/run:/var/run  rancher/rancher-agent:v2.8.5 --server https://10.0.0.76 --token tplxzwdhqjpr6vtc6jkq86zhcnftsqpq4t9j8ljjp4rxft6npxwr8g --ca-checksum 757201df237a2d92f909abb42db07c929d1153ec224869f5868211de191e0051 --worker --node-name $ip
fi