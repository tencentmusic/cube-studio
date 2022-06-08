# 所需要的所有镜像
kubeflow = [
    'gcr.io/kubeflow-images-public/xgboost-operator:vmaster-g56c2c075',
    'gcr.io/ml-pipeline/metadata-writer:1.0.4', 'gcr.io/tfx-oss-public/ml_metadata_store_server:v0.21.1',
    'gcr.io/ml-pipeline/envoy:metadata-grpc', 'mysql:8.0.3',
    'docker.io/kubeflowkatib/katib-db-manager:v1beta1-a96ff59',
    'docker.io/kubeflowkatib/katib-controller:v1beta1-a96ff59', 'argoproj/argoui:v2.3.0',
    'gcr.io/istio-release/proxy_init:release-1.3-latest-daily',
    'gcr.io/istio-release/kubectl:release-1.3-latest-daily', 'gcr.io/google_containers/spartakus-amd64:v1.1.0',
    'gcr.io/istio-release/proxyv2:release-1.3-latest-daily', 'mpioperator/mpi-operator:latest',
    'gcr.io/kubeflow-images-public/admission-webhook:vmaster-ge5452b6f',
    'gcr.io/kubeflow-images-public/tf_operator:vmaster-gda226016', 'istio/proxyv2:1.3.1',
    'gcr.io/istio-release/galley:release-1.3-latest-daily', 'quay.io/jetstack/cert-manager-cainjector:v0.11.0',
    'gcr.io/istio-release/citadel:release-1.3-latest-daily',
    'gcr.io/kubeflow-images-public/jupyter-web-app:vmaster-g845af298', 'python:3.7',
    'gcr.io/istio-release/mixer:release-1.3-latest-daily', 'gcr.io/istio-release/pilot:release-1.3-latest-daily',
    'gcr.io/spark-operator/spark-operator:v1beta2-1.1.0-2.4.5', 'gcr.io/kubebuilder/kube-rbac-proxy:v0.4.0',
    'gcr.io/tfx-oss-public/ml_metadata_store_server:0.22.1',
    'gcr.io/tfx-oss-public/ml_metadata_store_server:0.25.1',
    'gcr.io/istio-release/sidecar_injector:release-1.3-latest-daily',
    'quay.io/jetstack/cert-manager-webhook:v0.11.0',
    'gcr.io/kubeflow-images-public/kubernetes-sigs/application:1.0-beta',
    'gcr.io/kubeflow-images-public/centraldashboard:vmaster-g8097cfeb',
    'gcr.io/kubeflow-images-public/xgboost-operator:v0.1.0', 'quay.io/jetstack/cert-manager-controller:v0.11.0',
    'mysql:8', 'seldonio/seldon-core-operator:1.4.0', 'gcr.io/kfserving/kfserving-controller:v0.4.1',
    'gcr.io/istio-release/node-agent-k8s:release-1.3-latest-daily',
    'gcr.io/kubeflow-images-public/notebook-controller:vmaster-g6eb007d0',
    'gcr.io/kubeflow-images-public/pytorch-operator:vmaster-g518f9c76',
    'gcr.io/tfx-oss-public/ml_metadata_store_server:v0.21.1', 'metacontroller/metacontroller:v0.3.0',
    'prom/prometheus:v2.8.0', 'gcr.io/kubeflow-images-public/kfam:vmaster-g9f3bfd00',
    'kubeflow/mxnet-operator:v1.0.0-20200625',
    'gcr.io/kubeflow-images-public/profile-controller:vmaster-ga49f658f',
    'gcr.io/kubeflow-images-public/ingress-setup:latest',
    'alpine:3.10',
    "busybox"
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


new_katib = ['docker.io/kubeflowkatib/katib-ui:v1beta1-a96ff59',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/katib-db-manager', 'mysql:5.7','mysql:8.0.26',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/file-metrics-collector',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/katib-ui',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/katib-controller', 'docker.io/kubeflowkatib/mxnet-mnist',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/suggestion-hyperopt',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/tfevent-metrics-collector',
             'gcr.io/kubeflow-ci/pytorch-dist-mnist-test:v1.0',
             'gcr.io/kubeflow-images-public/katib/v1alpha3/suggestion-chocolate']

new_gpu = ['nvidia/k8s-device-plugin:v0.7.1', 'nvidia/dcgm-exporter:2.0.13-2.1.2-ubuntu20.04',
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
    'quay.io/prometheus-operator/prometheus-operator:v0.56.1',
    "k8s.gcr.io/prometheus-adapter/prometheus-adapter:v0.9.1",
    'grafana/grafana:7.5.2'
]
new_serving = ['gcr.io/kfserving/alibi-explainer:0.2.2', 'gcr.io/kfserving/logger:0.2.2', 'tensorflow/serving:1.14.0',
               'tensorflow/serving:1.14.0-gpu', 'tensorflow/serving:1.11.0', 'tensorflow/serving:1.11.0-gpu',
               'tensorflow/serving:1.12.0', 'tensorflow/serving:1.12.0-gpu', 'tensorflow/serving:1.13.0',
               'tensorflow/serving:1.13.0-gpu', 'tensorflow/serving:1.14.0', 'tensorflow/serving:1.14.0-gpu',
               'tensorflow/serving:2.0.0', 'tensorflow/serving:2.0.0-gpu', 'tensorflow/serving:2.1.0',
               'tensorflow/serving:2.1.0-gpu', 'tensorflow/serving:2.2.0', 'tensorflow/serving:2.2.0-gpu',
               'tensorflow/serving:2.3.0', 'tensorflow/serving:2.3.0-gpu', 'tensorflow/serving:2.4.0',
               'tensorflow/serving:2.4.0-gpu', 'mcr.microsoft.com/onnxruntime/server:v0.5.1',
               'gcr.io/kfserving/sklearnserver:0.2.2', 'gcr.io/kfserving/xgbserver:0.2.2',
               'gcr.io/kfserving/pytorchserver:0.2.2', 'nvcr.io/nvidia/tensorrtserver:19.05-py3',
               'gcr.io/kfserving/storage-initializer:0.2.2',
               'gcr.io/knative-releases/knative.dev/serving/cmd/queue:792f6945c7bc73a49a470a5b955c39c8bd174705743abf5fb71aa0f4c04128eb']

volcano = ['volcanosh/vc-controller-manager:v1.4.0', 'volcanosh/vc-scheduler:v1.4.0',
           'volcanosh/vc-webhook-manager:v1.4.0']

kube_batch = ['kubesigs/kube-batch:v0.5']
nni = ['frameworkcontrolle/frameworkcontrolle']


images = kubeflow + kubernetes_dashboard + new_pipline + new_gpu + new_prometheus + new_serving + volcano

# images = new_pipline
images = list(set(images))

# 通过私有仓库，将公有镜像下发到内网每台机器上，例如内网docker.oa.com的仓库
HOST = 'ccr.ccs.tencentyun.com/cube-studio/'
for image in images:
    # print(image)
    image = image.replace('<none>', '')
    image_name = HOST + image.replace(HOST,'').replace('/', '-').replace('@sha256', '')

    # 可联网机器上拉取公有镜像并推送到私有仓库
    # print('docker pull %s' % image)
    # print('docker tag %s %s' % (image, image_name))
    # print('docker push %s' % (image_name))

    # 内网机器上拉取私有仓库镜像
    image=image.replace('@sha256','')
    print("docker pull %s" % image_name)
    print("docker tag %s %s"%(image_name,image))







