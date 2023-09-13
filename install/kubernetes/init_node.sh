
mkdir -p /data/k8s/kubeflow/pipeline/workspace /data/k8s/kubeflow/pipeline/archives /data/k8s/infra/mysql
mkdir -p /data/k8s/monitoring/grafana/ /data/k8s/monitoring/prometheus/
chmod -R 777 /data/k8s/monitoring/grafana/ /data/k8s/monitoring/prometheus/
# 拉取镜像
docker login
sh pull_image_kubeflow.sh



