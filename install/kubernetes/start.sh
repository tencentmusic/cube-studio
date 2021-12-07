mkdir ~/.kube/
cp config ~/.kube/config
cp config kubeconfig/dev-kubeconfig
ip=`ifconfig eth1 | grep 'inet '| awk '{print $2}' | head -n 1`
kubectl label node $ip train=true cpu=true notebook=true service=true org=public istio=true knative=true kubeflow=true kubeflow-dashboard=true mysql=true redis=true monitoring=true logging=true
# 拉取镜像
sh pull_image_kubeflow.sh
curl -LO https://dl.k8s.io/release/v1.18.0/bin/linux/amd64/kubectl
chmod +x kubectl
mv kubectl /usr/bin/
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
kubectl create -f master.yaml
# 部署kube-batch
kubectl create -f kube-batch/deploy.yaml

# 部署prometheus
cd prometheus
chmod -R 777 /data/k8s/monitoring/grafana/
kubectl create -f ./operator/bundle.yaml
kubectl create -f ./alertmanater/alertmanager-main-sa.yml
kubectl create -f ./alertmanater/alertmanager-main-secret.yml
kubectl create -f ./alertmanater/alertmanager-main-svc.yml
kubectl create -f ./alertmanater/alertmanager-main.yml
kubectl create -f ./node-exporter/node-exporter-sa.yml
kubectl create -f ./node-exporter/node-exporter-rbac.yml
kubectl create -f ./node-exporter/node-exporter-svc.yml
kubectl create -f ./node-exporter/node-exporter-ds.yml
kubectl create -f ./kube-state-metrics/kube-state-metrics-sa.yml
kubectl create -f ./kube-state-metrics/kube-state-metrics-rbac.yml
kubectl create -f ./kube-state-metrics/kube-state-metrics-svc.yml
kubectl create -f ./kube-state-metrics/kube-state-metrics-dp.yml
kubectl create -f ./grafana/pv-pvc-hostpath.yml
kubectl create -f ./grafana/grafana-sa.yml
kubectl create -f ./grafana/grafana-source.yml
kubectl create -f ./grafana/grafana-datasources.yml
kubectl create -f ./grafana/grafana-admin-secret.yml
kubectl create -f ./grafana/grafana-svc.yml
kubectl create -f ./prometheus/prometheus-main.yml
kubectl create -f ./prometheus/prometheus-rules.yml
kubectl create -f ./prometheus/prometheus-rbac.yml
kubectl create -f ./prometheus/prometheus-svc.yml
kubectl create -f ./servicemonitor/alertmanager-sm.yml
kubectl create -f ./servicemonitor/coredns-sm.yml
kubectl create -f ./servicemonitor/kube-apiserver-sm.yml
kubectl create -f ./servicemonitor/kube-controller-manager-sm.yml
kubectl create -f ./servicemonitor/kube-scheduler-sm.yml
kubectl create -f ./servicemonitor/kubelet-sm.yml
kubectl create -f ./servicemonitor/kubestate-metrics-sm.yml
kubectl create -f ./servicemonitor/node-exporter-sm.yml
kubectl create -f ./servicemonitor/prometheus-operator-sm.yml
kubectl create -f ./servicemonitor/prometheus-sm.yml
kubectl create -f ./servicemonitor/pushgateway-sm.yml
cd ../


# 部署frameworkcontroller
cd frameworkcontroller
kubectl create serviceaccount frameworkcontroller --namespace service
kubectl create serviceaccount frameworkcontroller --namespace pipeline
kubectl create serviceaccount frameworkcontroller --namespace katib
kubectl create serviceaccount frameworkcontroller --namespace kubeflow
kubectl create clusterrolebinding frameworkcontroller-service --clusterrole=cluster-admin --user=system:serviceaccount:service:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-pipeline --clusterrole=cluster-admin --user=system:serviceaccount:pipeline:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-katib --clusterrole=cluster-admin --user=system:serviceaccount:katib:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-kubeflow --clusterrole=cluster-admin --user=system:serviceaccount:kubeflow:frameworkcontroller
kubectl create -f frameworkcontroller-with-default-config.yaml

kubectl create serviceaccount frameworkbarrier --namespace service
kubectl create serviceaccount frameworkbarrier --namespace pipeline
kubectl create serviceaccount frameworkbarrier --namespace katib
kubectl create serviceaccount frameworkbarrier --namespace kubeflow
kubectl create clusterrole frameworkbarrier --verb=get,list,watch --resource=frameworks
kubectl create clusterrolebinding frameworkbarrier-service --clusterrole=frameworkbarrier  --user=system:serviceaccount:service:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-pipeline --clusterrole=frameworkbarrier  --user=system:serviceaccount:pipeline:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-katib --clusterrole=frameworkbarrier  --user=system:serviceaccount:katib:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-kubeflow --clusterrole=frameworkbarrier  --user=system:serviceaccount:kubeflow:frameworkbarrier
cd ../

# 部署volcano
kubectl apply -f volcano/volcano-development.yaml


# 部署kubeflow
kubectl apply -f kubeflow/v1.2.0/sa-rbac.yaml
wget https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz && tar -zxvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
chmod +x kfctl
mv kfctl /usr/bin/
kfctl apply -V -f kubeflow/v1.2.0/kfctl_k8s_istio.v1.2.0.yaml

# 部署kfp pipeline
kubectl create -f kubeflow/pipeline/minio-pv-hostpath.yaml
kubectl apply -f kubeflow/pipeline/minio-artifact-secret.yaml
kubectl apply -f kubeflow/pipeline/pipeline-runner-rolebinding.yaml

kustomize build kubeflow/pipeline/1.6.0/kustomize/cluster-scoped-resources/ | kubectl apply -f -
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kustomize build kubeflow/pipeline/1.6.0/kustomize/env/platform-agnostic/  | kubectl apply -f -

# 部署xgb
kubectl kustomize  kubeflow/xgboost-operator/manifests/base | kubectl apply -f -


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

kubectl apply -k cube/overlays


kubectl apply -f gateway.yaml
kubectl apply -f sa-rbac.yaml
kubectl apply -f virtual.yaml


# 配置入口

kubectl patch svc istio-ingressgateway -n istio-system -p '{"spec":{"externalIPs":["'"${ip}"'"]}}'

# 本地电脑手动host
echo "在自己的电脑执行以下命令："
echo "sudo echo ${ip} kubeflow.local.com >> /etc/hosts"



