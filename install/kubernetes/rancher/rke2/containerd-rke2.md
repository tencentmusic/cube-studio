
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

k8s版本: 1.25， 不要部署nginx-ingress

镜像仓库：配置了这个，rancher的系统组件，就会走阿里云

    registry.cn-hangzhou.aliyuncs.com 

配置了这个，k8s里面拉取镜像就可以走镜像加速器

Mirror：

    镜像仓库主机名 *    mirror地址： docker.m.daocloud.io
    镜像仓库主机名 docker.io    mirror地址： docker.m.daocloud.io

高级选项-补充的 

```
kubelet参数：（使用主机的containerd）
container-runtime=remote
containerd=/run/containerd/containerd.sock
container-runtime-endpoint=unix:///run/containerd/containerd.sock

API server 参数： 
--service-node-port-range=1-65535

```



