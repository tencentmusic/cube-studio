#!/bin/bash

# 部署dashboard
kubectl delete -f dashboard/v2.2.0-cluster.yaml
# 高版本k8s部署2.6.1版本
kubectl delete -f dashboard/v2.6.1-cluster.yaml
# 部署mysql
kubectl delete -f mysql/deploy.yaml
kubectl delete -f mysql/service.yaml
kubectl delete -f mysql/configmap-mysql.yaml
kubectl delete -f mysql/pv-pvc-hostpath.yaml

# 部署redis
kubectl delete -f redis/master.yaml
kubectl delete -f redis/service.yaml
kubectl delete -f redis/pv-hostpath.yaml
kubectl delete -f redis/configmap.yaml

# 如果自己需要使用pv来保存redis队列数据，可以修改master.yaml

# 部署prometheus
cd prometheus
# 删除 prometheus_adapter
kubectl delete -f ./prometheus_adapter/prometheus_adapter.yaml
kubectl delete -f ./prometheus_adapter/metric_rule.yaml
# 卸载sm
kubectl delete -f ./servicemonitor/coredns-sm.yml
kubectl delete -f ./servicemonitor/kube-apiserver-sm.yml
kubectl delete -f ./servicemonitor/kube-controller-manager-sm.yml
kubectl delete -f ./servicemonitor/kube-scheduler-sm.yml
kubectl delete -f ./servicemonitor/kubelet-sm.yml
kubectl delete -f ./servicemonitor/kubestate-metrics-sm.yml
kubectl delete -f ./servicemonitor/node-exporter-sm.yml
kubectl delete -f ./servicemonitor/prometheus-operator-sm.yml
kubectl delete -f ./servicemonitor/prometheus-sm.yml
# 删除prometheus 实例
kubectl delete -f ./prometheus/prometheus-main.yml
kubectl delete -f ./prometheus/pv-pvc-hostpath.yaml
# 删除prometheus 配置
kubectl delete -f ./service-discovery/kube-controller-manager-svc.yml
kubectl delete -f ./service-discovery/kube-scheduler-svc.yml
kubectl delete -f ./prometheus/prometheus-secret.yml
kubectl delete -f ./prometheus/prometheus-rules.yml
kubectl delete -f ./prometheus/prometheus-rbac.yml
kubectl delete -f ./prometheus/prometheus-svc.yml
# 删除grafana
kubectl delete -f ./grafana/grafana-dp.yml
kubectl delete -f ./grafana/grafana-sa.yml
kubectl delete -f ./grafana/grafana-source.yml
kubectl delete -f ./grafana/grafana-datasources.yml
kubectl delete -f ./grafana/grafana-admin-secret.yml
kubectl delete -f ./grafana/grafana-svc.yml
kubectl delete configmap grafana-config all-grafana-dashboards --namespace=monitoring
kubectl delete -f ./grafana/pv-pvc-hostpath.yml
# 删除node exporter
kubectl delete -f ./node-exporter/node-exporter-sa.yml
kubectl delete -f ./node-exporter/node-exporter-rbac.yml
kubectl delete -f ./node-exporter/node-exporter-svc.yml
kubectl delete -f ./node-exporter/node-exporter-ds.yml
# 删除 crd operator
kubectl delete -f ./operator/operator-crd.yml
kubectl delete -f ./operator/operator-rbac.yml
kubectl delete -f ./operator/operator-dp.yml

cd ../


# 部署gpu的监控
kubectl delete -f gpu/nvidia-device-plugin.yml
kubectl delete -f gpu/dcgm-exporter.yaml

# 部署volcano
kubectl delete -f volcano/volcano-development.yaml
kubectl delete secret volcano-admission-secret -n kubeflow
# kubectl apply -f volcano/volcano-development.yaml
# kubectl wait crd/jobs.batch.volcano.sh --for condition=established --timeout=60s

# 部署istio
kubectl delete -f istio/install-crd.yaml
kubectl delete -f istio/install.yaml
kubectl delete -f istio/install-1.15.0.yaml

# kubectl wait crd/virtualservices.networking.istio.io --for condition=established --timeout=60s
# kubectl wait crd/gateways.networking.istio.io --for condition=established --timeout=60s

kubectl delete -f gateway.yaml
kubectl delete -f virtual.yaml

# 部署argo
kubectl delete -f argo/pipeline-runner-rolebinding.yaml
kubectl delete -f argo/install-3.4.3-all.yaml
kubectl delete -f argo/minio-pv-pvc-hostpath.yaml

# 部署trainjob:tfjob/pytorchjob/mpijob/mxnetjob/xgboostjobs
kubectl delete -f kubeflow/sa-rbac.yaml
kubectl delete -k kubeflow/train-operator/manifests/overlays/standalone

# 部署sparkjob
kubectl delete -f spark/install.yaml

# 部署paddlejob
kubectl apply -f paddle/crd.yaml
kubectl apply -f paddle/operator.yaml

# 部署管理平台
kubectl delete configmap kubernetes-config -n infra

kubectl delete configmap kubernetes-config -n pipeline

kubectl delete configmap kubernetes-config -n automl

kubectl delete -k cube/overlays
kubectl delete -f sa-rbac.yaml

kubectl delete -f pv-pvc-infra.yaml
kubectl delete -f pv-pvc-jupyter.yaml
kubectl delete -f pv-pvc-automl.yaml
kubectl delete -f pv-pvc-pipeline.yaml
kubectl delete -f pv-pvc-service.yaml

echo "ns未删除，可视情况删除ns"
