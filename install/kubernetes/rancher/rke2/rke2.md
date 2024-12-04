# 本文是测试文档，暂时未测试
```
apt install ipset ipvsadm -y

sudo modprobe ip_vs
sudo modprobe ip_vs_rr
sudo modprobe ip_vs_wrr
sudo modprobe ip_vs_lc
sudo modprobe ip_vs_wlc
sudo modprobe ip_vs_lblc
sudo modprobe ip_vs_lblcr
sudo modprobe ip_vs_sh
sudo modprobe ip_vs_dh
sudo modprobe ip_vs_sed
sudo modprobe ip_vs_nq


lsmod | grep ip_vs


cat > /etc/sysctl.d/k8s.conf << EOF
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
vm.swappiness = 0
EOF

modprobe br_netfilter
lsmod | grep br_netfilter
```

# rancher server 内自带的镜像
```bash
mkdir -p /data/rancher/k3s/agent/images/
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/k3s-airgap-images.tar -O /data/rancher/k3s/agent/images/k3s-airgap-images.tar

# docker run --rm --entrypoint "" -v $(pwd):/output rancher/rancher:v2.8.5 cp /var/lib/rancher/k3s/agent/images/k3s-airgap-images.tar /output/k3s-airgap-images.tar
# cp k3s-airgap-images.tar /data/rancher/k3s/agent/images/

```

# 修改rke2的containerd的配置文件
```bash
cp /var/lib/rancher/rke2/agent/etc/containerd/config.toml /var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl

tee -a /var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl << 'EOF'
[plugins."io.containerd.grpc.v1.cri".registry]
  config_path = "/etc/containerd/certs.d"
EOF

systemctl restart rke2-server
```


# rke2的使用

rke2的containerd实例在/var/lib/rancher/rke2/bin/
rke2的conatinerd实例的配置文件在/var/lib/rancher/rke2/agent/etc/containerd/config.toml
rke2的containerd实例通过rke2-server来管理，不需要自己管理，修改了配置的话 systemctl restart rke2-server，系统上的 containerd 可能是由 containerd.service 管理
rke2的conatinerd实例的socket在/run/k3s/containerd/containerd.sock，而系统上的 containerd 可能使用 /run/containerd/containerd.sock
rke2集群上启动的pod使用rke2自带的containerd实例，不用系统的，所以主机本身不需要安装containerd

# k8s的配置
```bash
export KUBECONFIG=/etc/rancher/rke2/rke2.yaml
/var/lib/rancher/rke2/bin/kubectl get nodes
/var/lib/rancher/rke2/bin/kubectl --kubeconfig /etc/rancher/rke2/rke2.yaml get nodes

```

# 查看rke2的containerd的配置
```bash
/var/lib/rancher/rke2/bin/containerd config dump
/run/k3s/containerd/containerd.sock

/var/lib/rancher/rke2/bin/ctr --address /run/k3s/containerd/containerd.sock --namespace k8s.io container ls
/var/lib/rancher/rke2/bin/ctr --address /run/k3s/containerd/containerd.sock --namespace k8s.io images ls

echo "export CRI_CONFIG_FILE=/var/lib/rancher/rke2/agent/etc/crictl.yaml" >> /etc/profile
source /etc/profile
cp /var/lib/rancher/rke2/bin/crictl /usr/local/bin/
crictl ps

/var/lib/rancher/rke2/bin/crictl --config /var/lib/rancher/rke2/agent/etc/crictl.yaml ps

/var/lib/rancher/rke2/bin/crictl --runtime-endpoint unix:///run/k3s/containerd/containerd.sock ps -a

systemctl restart rke2-server

```
