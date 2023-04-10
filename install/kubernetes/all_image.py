# 所需要的所有镜像
kubeflow = [
    'mysql:5.7',  # 数据库
    'bitnami/redis:4.0.14',  # 缓存
    'alpine:3.10',
    "busybox",
    "ccr.ccs.tencentyun.com/cube-studio/kubeflow:training-operator",  # 分布式训练
    'ccr.ccs.tencentyun.com/cube-studio/spark-operator:v1beta2-1.3.7-3.1.1',  # spark serverless
]

kubernetes_dashboard = [
    'kubernetesui/dashboard:v2.6.1',  # k8s dashboard
    'kubernetesui/metrics-scraper:v1.0.8',  # k8s dashboard 上的指标监控
]

new_gpu = [
    'nvidia/k8s-device-plugin:v0.7.1',  # gpu k8s插件
    'nvidia/dcgm-exporter:2.3.1-2.6.1-ubuntu20.04',  # gpu监控
    'tkestack/gpu-manager:1.0.3'
]

new_prometheus = [
    'quay.io/prometheus/alertmanager:v0.15.0',  # 报警
    'quay.io/prometheus-operator/prometheus-config-reloader:v0.46.0',  # prometheus配置翻译
    "quay.io/prometheus/prometheus:v2.27.1",  # peomethues数据库
    'quay.io/coreos/kube-state-metrics:v1.3.1',  # 状态 指标
    'quay.io/prometheus/node-exporter:v0.15.2',  # 机器指标
    'quay.io/coreos/kube-rbac-proxy:v0.3.1',  # 指标
    'quay.io/coreos/addon-resizer:1.0',  # 指标
    "quay.io/prometheus-operator/prometheus-operator:v0.46.0",  # prometheus 部署工具
    "k8s.gcr.io/prometheus-adapter/prometheus-adapter:v0.9.1",  # peometheus指标翻译为自定义指标
    'grafana/grafana:9.1.5'  # 监控看板
]

istio = [
    "istio/proxyv2:1.14.1",  # ingressgateway
    "istio/pilot:1.14.1"  # 数据面
]
volcano = [
    'volcanosh/vc-controller-manager:v1.4.0',  # 控制器
    'volcanosh/vc-scheduler:v1.4.0',  # 调度器
    'volcanosh/vc-webhook-manager:v1.4.0'  # 拦截器
]

nni = [
    'frameworkcontroller/frameworkcontroller'  # 超参搜索
]
pipeline = [
    'minio/minio',
    'quay.io/argoproj/argoexec:v3.4.3',
    'quay.io/argoproj/workflow-controller:latest',
    'quay.io/argoproj/workflow-controller:v3.4.3',
    'quay.io/argoproj/argocli:latest'
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
    "ubuntu:18.04",
    "ccr.ccs.tencentyun.com/cube-studio/datax:latest",
    "ccr.ccs.tencentyun.com/cube-studio/volcano:20211001",
    "ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20210601",
    "ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1",
    "ccr.ccs.tencentyun.com/cube-studio/xgb_train_and_predict:v1",
    "ccr.ccs.tencentyun.com/cube-studio/tf2.3_keras_train:latest",
    "ccr.ccs.tencentyun.com/cube-studio/tf2.3_plain_train:latest",
    "ccr.ccs.tencentyun.com/cube-studio/tf_distributed_train:latest",
    "ccr.ccs.tencentyun.com/cube-studio/tf2.3_model_evaluation:latest",
    "ccr.ccs.tencentyun.com/cube-studio/tf_distributed_eval:latest",
    "ccr.ccs.tencentyun.com/cube-studio/tf_model_offline_predict:latest",
    "ccr.ccs.tencentyun.com/cube-studio/pytorch_distributed_train_k8s:20201010",
    "ccr.ccs.tencentyun.com/cube-studio/horovod:20210401",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601",
    "ccr.ccs.tencentyun.com/cube-studio/kaldi_distributed_on_volcano:v2",
    "ccr.ccs.tencentyun.com/cube-studio/volcano:offline-predict-20220101",
    "ccr.ccs.tencentyun.com/cube-studio/object_detection_on_darknet:v1",
    "ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001"

    # 用户可能使用的基础镜像
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.0.3-cudnn8',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.0.3-cudnn8-python3.7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.0.3-cudnn8-python3.8',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.2-cudnn7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.2-cudnn7-python3.7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.2-cudnn7-python3.8',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7-python3.6',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7-python3.7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7-python3.8',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.0-cudnn7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.0-cudnn7-python3.6',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.0-cudnn7-python3.7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.0-cudnn7-python3.8',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda9.1-cudnn7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda9.1-cudnn7-python3.6',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda9.1-cudnn7-python3.7',
    'ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda9.1-cudnn7-python3.8',
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
    print("docker pull %s &" % (image,))

print('')
print('wait')




