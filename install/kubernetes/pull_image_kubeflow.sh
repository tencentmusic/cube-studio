docker pull busybox:1.36.0 &
docker pull nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 &
docker pull grafana/grafana:9.1.5 &
docker pull ccr.ccs.tencentyun.com/cube-studio/prometheus-adapter:v0.9.1 &
docker pull quay.io/prometheus-operator/prometheus-operator:v0.46.0 &
docker pull quay.io/prometheus-operator/prometheus-config-reloader:v0.46.0 &
docker pull volcanosh/vc-scheduler:v1.7.0 &
docker pull mysql:8.0.32 &
docker pull volcanosh/vc-controller-manager:v1.7.0 &
docker pull argoproj/workflow-controller:v3.4.3 &
docker pull carlosedp/addon-resizer:v1.8.4 &
docker pull istio/proxyv2:1.15.0 &
docker pull minio/minio:RELEASE.2023-04-20T17-56-55Z &
docker pull kubeflow/training-operator:v1-8a066f9 &
docker pull prom/prometheus:v2.27.1 &
docker pull argoproj/argocli:v3.4.3 &
docker pull volcanosh/vc-webhook-manager:v1.7.0 &
docker pull bitnami/redis:6.2.12 &
docker pull argoproj/argoexec:v3.4.3 &
docker pull bitnami/kube-rbac-proxy:0.14.1 &
docker pull kubernetesui/metrics-scraper:v1.0.8 &
docker pull nvidia/k8s-device-plugin:v0.11.0-ubuntu20.04 &
docker pull prom/node-exporter:v1.5.0 &
docker pull istio/pilot:1.15.0 &
docker pull kubernetesui/dashboard:v2.6.1 &

wait
