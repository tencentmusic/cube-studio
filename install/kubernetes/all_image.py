# 所需要的所有镜像
kubeflow = [
    'mysql:5.7',
    'bitnami/redis',
    'metacontroller/metacontroller:v0.3.0',
    'alpine:3.10',
    "busybox",
    "ccr.ccs.tencentyun.com/cube-studio/kubeflow:training-operator",
    'ccr.ccs.tencentyun.com/cube-studio/spark-operator:v1beta2-1.3.7-3.1.1',
]

kubernetes_dashboard=['kubernetesui/dashboard:v2.2.0','kubernetesui/metrics-scraper:v1.0.6','quay.io/kubernetes-ingress-controller/nginx-ingress-controller:0.30.0']

new_pipline = [
    'gcr.io/ml-pipeline/api-server:1.6.0',
    'gcr.io/ml-pipeline/viewer-crd-controller:1.6.0',
    'gcr.io/ml-pipeline/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance',
    'gcr.io/ml-pipeline/workflow-controller:v2.12.9-license-compliance',
    'gcr.io/ml-pipeline/frontend:1.6.0',
    'gcr.io/ml-pipeline/scheduledworkflow:1.6.0',
    'gcr.io/ml-pipeline/persistenceagent:1.6.0',
    'gcr.io/ml-pipeline/visualization-server:1.6.0',
    'gcr.io/ml-pipeline/metadata-envoy:1.6.0',
    'gcr.io/ml-pipeline/metadata-writer:1.6.0',
    'gcr.io/tfx-oss-public/ml_metadata_store_server:0.30.0',
    "gcr.io/ml-pipeline/argoexec:v2.7.5-license-compliance",
    "gcr.io/ml-pipeline/argoexec:v2.12.9-license-compliance"
]


new_gpu = ['nvidia/k8s-device-plugin:v0.7.1','nvidia/dcgm-exporter:2.3.1-2.6.1-ubuntu20.04',
           'nvidia/pod-gpu-metrics-exporter:v1.0.0-alpha']

new_prometheus = [
    'quay.io/prometheus/alertmanager:v0.15.0',
    'quay.io/prometheus-operator/prometheus-config-reloader:v0.46.0',
    'quay.io/prometheus-operator/prometheus-config-reloader:v0.48.0',
    'quay.io/prometheus-operator/prometheus-config-reloader:v0.56.1',
    'quay.io/prometheus/prometheus:v2.3.1',
    "quay.io/prometheus/prometheus:v2.27.1",
    'quay.io/coreos/kube-state-metrics:v1.3.1',
    'quay.io/prometheus/node-exporter:v0.15.2',
    'quay.io/coreos/kube-rbac-proxy:v0.3.1',
    'quay.io/coreos/addon-resizer:1.0',
    "quay.io/prometheus-operator/prometheus-operator:v0.46.0",
    'quay.io/prometheus-operator/prometheus-operator:v0.56.1',
    "k8s.gcr.io/prometheus-adapter/prometheus-adapter:v0.9.1",
    'grafana/grafana:7.5.2'
]

istio=[
    "istio/proxyv2:1.14.1","istio/pilot:1.14.1"
]
volcano = ['volcanosh/vc-controller-manager:v1.4.0', 'volcanosh/vc-scheduler:v1.4.0',
           'volcanosh/vc-webhook-manager:v1.4.0']

kube_batch = ['kubesigs/kube-batch:v0.5']
nni = ['frameworkcontroller/frameworkcontroller']

cube_studio = [
    # 平台构建的镜像
    'ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-cpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:vscode-ubuntu-gpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-gpu-base',
    'ccr.ccs.tencentyun.com/cube-studio/notebook:jupyter-ubuntu-cpu-1.0.0',
    'ccr.ccs.tencentyun.com/cube-studio/nni:20211003',
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
    
    # 用户可能使用的镜像
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

# images = kubeflow + kubernetes_dashboard + new_pipline + new_gpu + new_prometheus + volcano + kube_batch + nni+ cube_studio
images = kubeflow + kubernetes_dashboard + new_pipline + new_gpu + new_prometheus + volcano + kube_batch + nni
images = list(set(images))

# 通过私有仓库，将公有镜像下发到内网每台机器上，例如内网docker.oa.com的仓库
HOST = 'ccr.ccs.tencentyun.com/cube-studio/'
for image in images:
    # print(image)
    if 'gpu' in image:
        continue

    image = image.replace('<none>', '')
    image_name = HOST + image.replace(HOST,'').replace('/', '-').replace('@sha256', '')

    # 可联网机器上拉取公有镜像并推送到私有仓库
    # print('docker pull %s' % image)
    # print('docker tag %s %s' % (image, image_name))
    # print('docker push %s' % (image_name))

    # 内网机器上拉取私有仓库镜像
    # image=image.replace('@sha256','')
    # print("docker pull %s" % image_name)
    # print("docker tag %s %s"%(image_name,image))

    image=image.replace('@sha256','')
    print("docker pull %s && docker tag %s %s &" % (image_name,image_name,image))


print('')
print('wait')




