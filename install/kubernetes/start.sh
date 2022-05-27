
mkdir -p ~/.kube/ kubeconfig /data/k8s/kubeflow/pipeline/workspace /data/k8s/kubeflow/pipeline/archives
cp config ~/.kube/config
cp config kubeconfig/dev-kubeconfig

node=`kubectl  get node -o wide |grep $1 |awk '{print $1}'| head -n 1`
kubectl label node $node train=true cpu=true notebook=true service=true org=public istio=true knative=true kubeflow=true kubeflow-dashboard=true mysql=true redis=true monitoring=true logging=true --overwrite
# 拉取镜像
sh pull_image_kubeflow.sh
curl -LO https://dl.k8s.io/release/v1.18.0/bin/linux/amd64/kubectl && chmod +x kubectl  && mv kubectl /usr/bin/
wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv4.5.1/kustomize_v4.5.1_linux_amd64.tar.gz && tar -zxvf kustomize_v4.5.1_linux_amd64.tar.gz && chmod +x kustomize && mv kustomize /usr/bin/

# 创建命名空间
sh create_ns_secret.sh
# 部署dashboard
kubectl apply -f dashboard/v2.2.0-cluster.yaml
kubectl apply -f dashboard/v2.2.0-user.yaml
# 部署mysql
kubectl create -f mysql/pv-pvc-hostpath.yaml
kubectl create -f mysql/service.yaml
kubectl create -f mysql/configmap-mysql.yaml
kubectl create -f mysql/deploy.yaml
# 部署redis
kubectl create -f redis/pv-hostpath.yaml
kubectl create -f redis/configmap.yaml
kubectl create -f redis/service.yaml
# 如果自己需要使用pv来保存redis队列数据，可以修改master.yaml
kubectl create -f redis/master.yaml
# 部署kube-batch
kubectl create -f kube-batch/deploy.yaml

# 部署prometheus
cd prometheus
mkdir -p /data/k8s/monitoring/grafana/
chmod -R 777 /data/k8s/monitoring/grafana/
kubectl apply -f ./operator/bundle.yaml
kubectl apply -f ./alertmanater/alertmanager-main-sa.yml
kubectl apply -f ./alertmanater/alertmanager-main-secret.yml
kubectl apply -f ./alertmanater/alertmanager-main-svc.yml
kubectl apply -f ./alertmanater/alertmanager-main.yml
kubectl apply -f ./node-exporter/node-exporter-sa.yml
kubectl apply -f ./node-exporter/node-exporter-rbac.yml
kubectl apply -f ./node-exporter/node-exporter-svc.yml
kubectl apply -f ./node-exporter/node-exporter-ds.yml
kubectl apply -f ./kube-state-metrics/kube-state-metrics-sa.yml
kubectl apply -f ./kube-state-metrics/kube-state-metrics-rbac.yml
kubectl apply -f ./kube-state-metrics/kube-state-metrics-svc.yml
kubectl apply -f ./kube-state-metrics/kube-state-metrics-dp.yml
kubectl apply -f ./grafana/pv-pvc-hostpath.yml
kubectl apply -f ./grafana/grafana-sa.yml
kubectl apply -f ./grafana/grafana-source.yml
kubectl apply -f ./grafana/grafana-datasources.yml
kubectl apply -f ./grafana/grafana-admin-secret.yml
kubectl apply -f ./grafana/grafana-svc.yml
kubectl create configmap grafana-config --from-file=./grafana/grafana.ini --namespace=monitoring
kubectl create configmap all-grafana-dashboards --from-file=./grafana/dashboard --namespace=monitoring
kubectl apply -f ./grafana/grafana-dp.yml
kubectl apply -f ./service-discovery/kube-controller-manager-svc.yml
kubectl apply -f ./service-discovery/kube-scheduler-svc.yml
kubectl apply -f ./prometheus/prometheus-secret.yml
kubectl apply -f ./prometheus/prometheus-rules.yml
kubectl apply -f ./prometheus/prometheus-rbac.yml
kubectl apply -f ./prometheus/prometheus-svc.yml
kubectl apply -f ./prometheus/prometheus-main.yml
kubectl apply -f ./servicemonitor/alertmanager-sm.yml
kubectl apply -f ./servicemonitor/coredns-sm.yml
kubectl apply -f ./servicemonitor/kube-apiserver-sm.yml
kubectl apply -f ./servicemonitor/kube-controller-manager-sm.yml
kubectl apply -f ./servicemonitor/kube-scheduler-sm.yml
kubectl apply -f ./servicemonitor/kubelet-sm.yml
kubectl apply -f ./servicemonitor/kubestate-metrics-sm.yml
kubectl apply -f ./servicemonitor/node-exporter-sm.yml
kubectl apply -f ./servicemonitor/prometheus-operator-sm.yml
kubectl apply -f ./servicemonitor/prometheus-sm.yml
kubectl apply -f ./servicemonitor/pushgateway-sm.yml
kubectl apply -f ./prometheus_adapter/metric_rule.yaml
kubectl apply -f ./prometheus_adapter/prometheus_adapter.yaml
cd ../

# 部署gpu的监控
kubectl apply -f gpu/nvidia-device-plugin.yml
kubectl apply -f gpu/dcgm-exporter.yaml
kubectl apply -f gpu/dcgm-exporter-sm.yaml

# 部署frameworkcontroller

kubectl create serviceaccount frameworkcontroller --namespace service
kubectl create serviceaccount frameworkcontroller --namespace pipeline
kubectl create serviceaccount frameworkcontroller --namespace katib
kubectl create serviceaccount frameworkcontroller --namespace kubeflow
kubectl create clusterrolebinding frameworkcontroller-service --clusterrole=cluster-admin --user=system:serviceaccount:service:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-pipeline --clusterrole=cluster-admin --user=system:serviceaccount:pipeline:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-katib --clusterrole=cluster-admin --user=system:serviceaccount:katib:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-kubeflow --clusterrole=cluster-admin --user=system:serviceaccount:kubeflow:frameworkcontroller
kubectl create -f frameworkcontroller/frameworkcontroller-with-default-config.yaml

kubectl create serviceaccount frameworkbarrier --namespace service
kubectl create serviceaccount frameworkbarrier --namespace pipeline
kubectl create serviceaccount frameworkbarrier --namespace katib
kubectl create serviceaccount frameworkbarrier --namespace kubeflow
kubectl create clusterrole frameworkbarrier --verb=get,list,watch --resource=frameworks
kubectl create clusterrolebinding frameworkbarrier-service --clusterrole=frameworkbarrier  --user=system:serviceaccount:service:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-pipeline --clusterrole=frameworkbarrier  --user=system:serviceaccount:pipeline:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-katib --clusterrole=frameworkbarrier  --user=system:serviceaccount:katib:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-kubeflow --clusterrole=frameworkbarrier  --user=system:serviceaccount:kubeflow:frameworkbarrier


# 部署volcano
kubectl delete -f volcano/volcano-development.yaml
kubectl apply -f volcano/volcano-development.yaml
kubectl wait crd/jobs.batch.volcano.sh --for condition=established --timeout=60s

# 部署kubeflow(训练框架+istio)
kubectl apply -f kubeflow/v1.2.0/sa-rbac.yaml
#wget https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz && tar -zxvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
wget https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz && tar -zxvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
chmod +x kfctl
mv kfctl /usr/bin/
kfctl apply -V -f kubeflow/v1.2.0/kfctl_k8s_istio.v1.2.0.yaml

kubectl delete apiservice v1beta1.webhook.cert-manager.io

# 部署kfp pipeline
kubectl create -f kubeflow/pipeline/minio-pv-hostpath.yaml
kubectl apply -f kubeflow/pipeline/minio-artifact-secret.yaml
kubectl apply -f kubeflow/pipeline/pipeline-runner-rolebinding.yaml

cd kubeflow/pipeline/1.6.0/kustomize/
kustomize build cluster-scoped-resources/ | kubectl apply -f -
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kustomize build env/platform-agnostic/  | kubectl apply -f -
cd ../../../../

# 部署xgb
#kubectl kustomize  kubeflow/xgboost-operator/manifests/base | kubectl apply -f -


# 部署管理平台
kubectl delete configmap kubernetes-config -n infra
kubectl create configmap kubernetes-config --from-file=kubeconfig -n infra

kubectl delete configmap kubernetes-config -n pipeline
kubectl create configmap kubernetes-config --from-file=kubeconfig -n pipeline

kubectl delete configmap kubernetes-config -n katib
kubectl create configmap kubernetes-config --from-file=kubeconfig -n katib

kubectl create -f pv-pvc-infra.yaml
kubectl create -f pv-pvc-jupyter.yaml
kubectl create -f pv-pvc-katib.yaml
kubectl create -f pv-pvc-pipeline.yaml
kubectl create -f pv-pvc-service.yaml

kubectl delete -k cube/overlays
kubectl apply -k cube/overlays

kubectl wait crd/virtualservices.networking.istio.io --for condition=established --timeout=60s
kubectl wait crd/gateways.networking.istio.io --for condition=established --timeout=60s

kubectl apply -f gateway.yaml
kubectl apply -f sa-rbac.yaml
kubectl apply -f virtual.yaml


# 配置入口
#ip=`ifconfig eth1 | grep 'inet '| awk '{print $2}' | head -n 1`
kubectl patch svc istio-ingressgateway -n istio-system -p '{"spec":{"externalIPs":["'"$1"'"]}}'

# 本地电脑手动host
echo "打开网址：http://$1"



