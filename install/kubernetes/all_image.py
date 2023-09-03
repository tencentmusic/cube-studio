# 所需要的所有镜像
kubeflow = [
    'mysql:8.0.32',  # 数据库
    'bitnami/redis:6.2.12',  # 缓存
    "busybox:1.36.0",
    "kubeflow/training-operator:v1-8a066f9",  # 分布式训练
]

kubernetes_dashboard = [
    'kubernetesui/dashboard:v2.6.1',  # k8s dashboard
    'kubernetesui/metrics-scraper:v1.0.8',  # k8s dashboard 上的指标监控
]

new_gpu = [
    'nvidia/k8s-device-plugin:v0.11.0-ubuntu20.04',  # gpu k8s插件
    'nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04',  # gpu监控
]

new_prometheus = [
    "prom/prometheus:v2.27.1",  # peomethues数据库
    'prom/node-exporter:v1.5.0',  # 机器指标

    'quay.io/prometheus-operator/prometheus-config-reloader:v0.46.0',  # prometheus配置翻译
    "quay.io/prometheus-operator/prometheus-operator:v0.46.0",  # prometheus 部署工具
    'bitnami/kube-rbac-proxy:0.14.1',  # 指标
    'carlosedp/addon-resizer:v1.8.4',  # 指标

    'grafana/grafana:9.1.5'  # 监控看板
    "ccr.ccs.tencentyun.com/cube-studio/prometheus-adapter:v0.9.1",  # peometheus指标翻译为自定义指标
]

istio = [
    "istio/proxyv2:1.15.0",  # ingressgateway
    "istio/pilot:1.15.0"  # 数据面
]
volcano = [
    'volcanosh/vc-controller-manager:v1.7.0',  # 控制器
    'volcanosh/vc-scheduler:v1.7.0',  # 调度器
    'volcanosh/vc-webhook-manager:v1.7.0'  # 拦截器
]

nni = [
    'frameworkcontroller/frameworkcontroller'  # 超参搜索
]
pipeline = [
    'minio/minio:RELEASE.2023-04-20T17-56-55Z',
    'argoproj/argoexec:v3.4.3',
    'argoproj/workflow-controller:v3.4.3',
    'argoproj/argocli:v3.4.3'
]
cube_studio = [
    # notebook基础镜像
    'ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-cpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-gpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-gpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-1.0.0',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-bigdata',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-machinelearning',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-deeplearning',
    # 超参搜索的镜像
    'ccr.ccs.tencentyun.com/cube-studio/nni:20211003',

    # 推理服务的镜像
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:1.14.0',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:1.14.0-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.0.0',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.0.0-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.1.4',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.1.4-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.2.3',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.2.3-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.3.4',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.3.4-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.4.3',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.4.3-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.5.2',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.5.2-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.6.0',
    'ccr.ccs.tencentyun.com/cube-studio/tfserving:2.6.0-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/tritonserver:21.12-py3',
    'ccr.ccs.tencentyun.com/cube-studio/tritonserver:21.09-py3',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.6.0-cpu',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.6.0-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.5.3-cpu',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.5.3-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.4.2-cpu',
    'ccr.ccs.tencentyun.com/cube-studio/torchserve:0.4.2-gpu',
    'ccr.ccs.tencentyun.com/cube-studio/onnxruntime:latest',
    'ccr.ccs.tencentyun.com/cube-studio/onnxruntime:latest-cuda',

    # 任务模板的镜像
    "ubuntu:20.04",
    'python:3.9',
    "ccr.ccs.tencentyun.com/cube-studio/datax:latest",
    "ccr.ccs.tencentyun.com/cube-studio/volcano:20211001",
    "ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20210601",
    "ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1",
    "ccr.ccs.tencentyun.com/cube-studio/xgb:20230801",
    "ccr.ccs.tencentyun.com/cube-studio/pytorch:20201010",
    "ccr.ccs.tencentyun.com/cube-studio/horovod:20210401",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/kaldi_distributed_on_volcano:v2",
    "ccr.ccs.tencentyun.com/cube-studio/volcano:offline-predict-20220101",
    "ccr.ccs.tencentyun.com/cube-studio/object_detection_on_darknet:v1",
    "ccr.ccs.tencentyun.com/cube-studio/deploy-service:20250501",

    # 用户可能使用的基础镜像
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9',

]

# images = kubeflow + kubernetes_dashboard + new_gpu + new_prometheus + istio+ volcano + nni+ pipeline+cube_studio
images = kubeflow + kubernetes_dashboard + new_gpu + new_prometheus + istio + volcano + nni + pipeline
images = list(set(images))

# 通过私有仓库，将公有镜像下发到内网每台机器上，例如内网docker.oa.com的仓库
HOST = 'ccr.ccs.tencentyun.com/cube-studio/'
for image in images:
    # print(image)
    # print(image)
    image = image.replace('<none>', '')
    image_name = HOST + image.replace(HOST, '').replace('/', '-').replace('@sha256', '')

    # 可联网机器上拉取公有镜像并推送到私有仓库
    # print('docker pull %s && docker tag %s %s && docker push %s &' % (image,image,image_name,image_name))

    # # # 内网机器上拉取私有仓库镜像
    # image=image.replace('@sha256','')
    # print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))

    # # 拉取公有镜像
    image = image.replace('@sha256', '')
    # print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))
    print("docker pull %s &" % (image,))


print('')
print('wait')




