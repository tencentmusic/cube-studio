
# 1、gpu机器环境的准备  

安装gpu驱动
https://www.nvidia.cn/Download/index.aspx?lang=cn
安装cuda
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network

安装nvidia-docker2


首先需要找运维同学安装机器gpu卡对应的驱动，然后需要让你的docker能识别并应用gpu驱动。

 - 如果你的docker是19.03及以后的版本，并且只在docker中使用而不在k8s中使用，可以只安装nvidia-container-runtime 或者 只安装nvidia-container-toolkit，然后重启docker，就可以在docker run时通过添加参数--gpus 来应用gpu卡了。

 - 如果你的docker是19.03以前的版本，或者19.03以后的版本并需要在k8s中使用gpu，那需要安装nvidia docker2，因为k8s还没有支持docker的--gpu参数。安装nvidia docker2以后，修改docker 默认runtime。重启docker，这样就能在docker或者k8s中使用gpu了。
```
cat /etc/docker/daemon.json

{
    "insecure-registries":["docker.oa.com:8080"],
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
总结下：

1、找运维同学在机器上安装gpu驱动  

2、安装nvidia docker2（k8s没有支持新版本docker的--gpu）  

3、修改docker配置文件  

# 2、将机器加入k8s集群

加入k8s集群后为机器添加标签，标签只是用户管理和选择机型设备。
```
gpu=true       用于表示 gpu设备
gpu-type=V100  用于表示gpu型号，或者gpu-type=T4

train=true     用于训练
service=true   用于推理
notebook=true  用于开发
```

# 3、部署k8s gpu插件(vgpu)

daemonset 	kube-system/nvidia-device-plugin.会在机器上部署pod，用于scheduler识别改机器可用gpu算力。

daemonset 	kube-system/gpu-manager,会在gpu上虚拟化多张卡，在plugin中不同的虚拟化方式，有可能会占用的方式不同。此处使用的虚拟化方式不影响调用方式


使用vgpu添加的挂载，
```bash
/var/lib/kubelet/device-plugins:/var/lib/kubelet/device-plugins
/etc/gpu-manager/vm:/etc/gpu-manager/vm
/etc/gpu-manager/vdriver:/etc/gpu-manager/vdriver
/var/run/docker.sock:/var/run/docker.sock
/sys/fs/cgroup:/sys/fs/cgroup
/usr:/usr
```


# 4、部署k8s监控组件
daemonset 	monitoring/dcgm-exporter.会在机器上部署pod，用于监控gpu上的使用率

ServiceMonitor 	monitoring/dcgm-exporter-sm.用户暴露gpu指标给prometheus

# 5、调度占用gpu
直接写占用卡数目，对于异构gpu环境，也可以选择占用的卡型。比如1(T4),2(V100),1(VGPU)

# 6、基于gpu利用率进行弹性伸缩
需要先将gpu指标采集到prometheus，然后再将prometheus指标转变为对象指标控制hpa。

prometheus/prometheus_adapter目录下会部署kube-system/kubeflow-prometheus-adapter用于将prometheus指标转变为对象指标
