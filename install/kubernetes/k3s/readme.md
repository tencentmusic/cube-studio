# master节点
```bash
# 关闭防火墙
systemctl stop firewalld && systemctl disable firewalld && iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X
# 下载部署脚本
git clone -b v1.24.7+k3s1 https://github.com/k3s-io/k3s.git
cd k3s
# 设置版本
export INSTALL_K3S_VERSION=v1.24.7+k3s1
# 设置k8s部署配置
#export INSTALL_K3S_EXEC="--system-default-registry registry.cn-hangzhou.aliyuncs.com --write-kubeconfig ~/.kube/config --disable=traefik --cluster-cidr  10.72.0.0/16 --service-cidr  10.73.0.0/16"
export INSTALL_K3S_EXEC="--system-default-registry registry.cn-hangzhou.aliyuncs.com --write-kubeconfig ~/.kube/config --disable=traefik"
# 设置使用国内源
export INSTALL_K3S_MIRROR=cn
# 设置强制下载
export INSTALL_K3S_SYMLINK=force
#export INSTALL_K3S_FORCE_RESTART=true
# 设置镜像url
export INSTALL_K3S_MIRROR_URL=${INSTALL_K3S_MIRROR_URL:-'rancher-mirror.rancher.cn'}
# 替换github和storage 国内可以链接到的网络
export GITHUB_URL=https://githubfast.com/k3s-io/k3s/releases
export STORAGE_URL=https://k3s-ci-builds.s3.amazonaws.com
sed -i 's|^GITHUB_URL=.*|GITHUB_URL=https://githubfast.com/k3s-io/k3s/releases|' install.sh
sed -i 's|^STORAGE_URL=.*|STORAGE_URL=https://k3s-ci-builds.s3.amazonaws.com|' install.sh
# 部署
sh install.sh
# 打印master的token
cat /var/lib/rancher/k3s/server/node-token

# 设置 containerd 的 mirror
cat > /etc/rancher/k3s/registries.yaml <<EOF
mirrors:
  docker.io:
    endpoint:
      - "http://hub-mirror.c.163.com"
      - "https://docker.mirrors.ustc.edu.cn"
      - "https://registry.docker-cn.com"
EOF

```

# worker节点
```bash
systemctl stop firewalld && systemctl disable firewalld && iptables -F && iptables -t nat -F && iptables -t mangle -F && iptables -X
git clone -b v1.24.7+k3s1 https://github.com/k3s-io/k3s.git
cd k3s
export INSTALL_K3S_VERSION=v1.24.7
export K3S_URL=https://myserver:6443
export K3S_TOKEN=XXX
sh install.sh
```

# 清理
```bash
/usr/local/bin/k3s-killall.sh
/usr/local/bin/k3s-uninstall.sh
```

# 重启
```bash
sudo systemctl stop k3s
sudo systemctl start k3s
sudo systemctl stop k3s-agent
sudo systemctl start k3s-agent
```
