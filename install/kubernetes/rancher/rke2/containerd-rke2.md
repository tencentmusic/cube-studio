
# 部署rancher server

```bash

mkdir -p /data/rancher/k3s/agent/images/
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/k3s-airgap-images.tar -O /data/rancher/k3s/agent/images/k3s-airgap-images.tar
nerdctl network create cube-studio

export RANCHER_CONTAINER_TAG=v2.8.5
export PASSWORD=cube-studio
nerdctl run -d --privileged --network cube-studio  --restart=unless-stopped -p 443:443 --name=myrancher -e AUDIT_LEVEL=3 -e CATTLE_SYSTEM_DEFAULT_REGISTRY=registry.cn-hangzhou.aliyuncs.com -e CATTLE_BOOTSTRAP_PASSWORD=$PASSWORD -v /data/rancher:/var/lib/rancher registry.cn-hangzhou.aliyuncs.com/rancher/rancher:$RANCHER_CONTAINER_TAG

# 打开 https://xx.xx.xx.xx:443/ 等待web界面可以打开。预计要1~10分钟
# 输入密码cube-studio
```

# 部署rke2

镜像仓库：registry.cn-hangzhou.aliyuncs.com 

高级选项-补充的 API server 参数： --service-node-port-range=1-65535


yaml配置，目前不使用
```bash

registries:
  mirrors:
    docker.io:
      endpoint:
        - https://registry.cn-hangzhou.aliyuncs.com
        
# kubelet参数暂时不能用
kubelet:
  extra_args:
    container-runtime: remote
    container-runtime-endpoint: unix:///run/containerd/containerd.sock
    containerd: /run/containerd/containerd.sock
    runtime-request-timeout: 15m

--container-runtime=remote
--container-runtime-endpoint=unix:///run/containerd/containerd.sock
--containerd=/run/containerd/containerd.sock

```

# 修改rke2的containerd的配置文件
rke2的k8s部署完成后，修改rke2的配置。
```bash
cp /var/lib/rancher/rke2/agent/etc/containerd/config.toml /var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl

tee -a /var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl << 'EOF'
[plugins."io.containerd.grpc.v1.cri".registry]
  config_path = "/etc/containerd/certs.d"
EOF

systemctl restart rke2-server
# 查看最新的配置是否使用 /etc/containerd/certs.d下的加速器配置了
cat /var/lib/rancher/rke2/agent/etc/containerd/config.toml
```

