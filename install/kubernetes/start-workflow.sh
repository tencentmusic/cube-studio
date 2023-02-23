
bash init_node.sh
iptables -P FORWARD ACCEPT
iptables -P INPUT ACCEPT
iptables -P OUTPUT ACCEPT
mkdir -p ~/.kube && cp config ~/.kube/config && cp ~/.kube/config /etc/kubernetes/admin.conf
mkdir -p kubeconfig && echo "" > kubeconfig/dev-kubeconfig
curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl && chmod +x kubectl  && cp kubectl /usr/bin/ && mv kubectl /usr/local/bin/
node=`kubectl  get node -o wide |grep $1 |awk '{print $1}'| head -n 1`
kubectl label node $node train=true cpu=true notebook=true service=true org=public istio=true kubeflow=true kubeflow-dashboard=true mysql=true redis=true monitoring=true logging=true --overwrite

# 创建命名空间
sh create_ns_secret.sh
kubectl apply -f sa-rbac.yaml
# 部署dashboard
kubectl apply -f dashboard/v2.2.0-cluster.yaml
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

# 部署istio
kubectl apply -f istio/install-crd.yaml
kubectl wait crd/envoyfilters.networking.istio.io --for condition=established --timeout=60s
kubectl apply -f istio/install.yaml
# k8s 1.21+
# kubectl delete -f istio/install.yaml
# kubectl apply -f istio/install-1.15.0.yaml

kubectl wait crd/virtualservices.networking.istio.io --for condition=established --timeout=60s
kubectl wait crd/gateways.networking.istio.io --for condition=established --timeout=60s

kubectl apply -f gateway.yaml
kubectl apply -f virtual.yaml

# 部署argo
kubectl apply -f argo/minio-pv-pvc-hostpath.yaml
kubectl apply -f argo/pipeline-runner-rolebinding.yaml
kubectl apply -f argo/install-3.4.3-all.yaml

# 部署管理平台
kubectl delete configmap kubernetes-config -n infra
kubectl create configmap kubernetes-config --from-file=kubeconfig -n infra

kubectl delete configmap kubernetes-config -n pipeline
kubectl create configmap kubernetes-config --from-file=kubeconfig -n pipeline

kubectl delete configmap kubernetes-config -n automl
kubectl create configmap kubernetes-config --from-file=kubeconfig -n automl

kubectl create -f pv-pvc-infra.yaml
kubectl create -f pv-pvc-jupyter.yaml
kubectl create -f pv-pvc-automl.yaml
kubectl create -f pv-pvc-pipeline.yaml
kubectl create -f pv-pvc-service.yaml

kubectl delete -k cube/overlays
kubectl apply -k cube/overlays

# 配置入口
#ip=`ifconfig eth1 | grep 'inet '| awk '{print $2}' | head -n 1`
kubectl patch svc istio-ingressgateway -n istio-system -p '{"spec":{"externalIPs":["'"$1"'"]}}'

# 本地电脑手动host
echo "打开网址：http://$1"



