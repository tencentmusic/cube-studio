为部署机器添加label 
```bash
kubectl label node xxx monitoring=true
```


创建命名空间
```bash
kubectl create ns monitoring
```


# 部署operator
```
kubectl apply -f ./operator/bundle.yaml
```


 # 创建alert的配置文件，定义报警方式
```bash
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

```

 # 自定义配置文件，定义显示方式
```
# 按照自己的分布式存储创建pv
kubectl apply -f ./grafana/pv-pvc-hostpath.yml
kubectl apply -f ./grafana/grafana-sa.yml
kubectl apply -f ./grafana/grafana-source.yml
kubectl apply -f ./grafana/grafana-datasources.yml
kubectl apply -f ./grafana/grafana-admin-secret.yml
kubectl apply -f ./grafana/grafana-svc.yml
```

 # 创建配置conifgmap
```
kubectl create configmap grafana-config --from-file=./grafana/grafana.ini --namespace=monitoring 
kubectl apply -f ./grafana/grafana-dp.yml
kubectl apply -f ./service-discovery/kube-controller-manager-svc.yml
kubectl apply -f ./service-discovery/kube-scheduler-svc.yml
```

 # 自定义配置文件，定义收集和报警规则
```bash
kubectl apply -f ./prometheus/prometheus-secret.yml
kubectl apply -f ./prometheus/prometheus-rules.yml
kubectl apply -f ./prometheus/prometheus-rbac.yml
kubectl apply -f ./prometheus/prometheus-svc.yml
```

# prometheus-operator  部署成功后才能创建成功  
```bash
kubectl apply -f ./prometheus/prometheus-main.yml
```

 # 监控目标，lable必须是k8s-app  因为prometheus是按这个查找的。不然prometheus采集不了该资源
```
# kubelet监控，请先确保每个节点的kubelet 添加了authentication-token-webhook=true和authorization-mode=Webhook参数
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

```

# 部署gpu监控
```
kubectl apply -f ./gpu/dcgm-exporter.yaml
kubectl apply -f ./gpu/dcgm-exporter-sm.yaml
```

# prometheus-adapter  部署
```bash
kubectl apply -f ./prometheus_adapter/metric_rule.yaml
kubectl apply -f ./prometheus_adapter/prometheus_adapter.yaml
```


