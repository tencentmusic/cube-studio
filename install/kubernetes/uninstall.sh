
#!/bin/bash

read -p "是否继续执行脚本？(yes/no)" answer
if [[ "$answer" == "yes" ]]; then
    # mkdir -p ~/.kube && cp config ~/.kube/config && cp ~/.kube/config /etc/kubernetes/admin.conf
    # mkdir -p kubeconfig && echo "" > kubeconfig/dev-kubeconfig
    # curl -LO https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl && chmod +x kubectl  && cp kubectl /usr/bin/ && mv kubectl /usr/local/bin/
    node=`kubectl  get node -o wide |grep $1 |awk '{print $1}'| head -n 1`
    # kubectl label node $node train=true cpu=true notebook=true service=true org=public istio=true kubeflow=true kubeflow-dashboard=true mysql=true redis=true monitoring=true logging=true --overwrite

    # 创建命名空间
    # sh create_ns_secret.sh
    
    # 部署dashboard
    kubectl delete -f dashboard/v2.2.0-cluster.yaml
    # 高版本k8s部署2.6.1版本
    kubectl delete -f dashboard/v2.6.1-cluster.yaml
    # 部署mysql
    kubectl delete -f mysql/pv-pvc-hostpath.yaml
    kubectl delete -f mysql/service.yaml
    kubectl delete -f mysql/configmap-mysql.yaml
    kubectl delete -f mysql/deploy.yaml
    # 部署redis
    kubectl delete -f redis/pv-hostpath.yaml
    kubectl delete -f redis/configmap.yaml
    kubectl delete -f redis/service.yaml
    # 如果自己需要使用pv来保存redis队列数据，可以修改master.yaml
    kubectl delete -f redis/master.yaml

    # 部署prometheus
    cd prometheus
    kubectl delete -f ./operator/operator-crd.yml
    sleep 5
    kubectl delete -f ./operator/operator-crd.yml
    kubectl delete -f ./operator/operator-rbac.yml
    # kubectl wait crd/podmonitors.monitoring.coreos.com --for condition=established --timeout=60s
    kubectl delete -f ./operator/operator-dp.yml
    kubectl delete -f ./alertmanater/alertmanager-main-sa.yml
    kubectl delete -f ./alertmanater/alertmanager-main-secret.yml
    kubectl delete -f ./alertmanater/alertmanager-main-svc.yml
    kubectl delete -f ./alertmanater/alertmanager-main.yml
    kubectl delete -f ./node-exporter/node-exporter-sa.yml
    kubectl delete -f ./node-exporter/node-exporter-rbac.yml
    kubectl delete -f ./node-exporter/node-exporter-svc.yml
    kubectl delete -f ./node-exporter/node-exporter-ds.yml
    kubectl delete -f ./kube-state-metrics/kube-state-metrics-sa.yml
    kubectl delete -f ./kube-state-metrics/kube-state-metrics-rbac.yml
    kubectl delete -f ./kube-state-metrics/kube-state-metrics-svc.yml
    kubectl delete -f ./kube-state-metrics/kube-state-metrics-dp.yml
    kubectl delete -f ./grafana/pv-pvc-hostpath.yml
    kubectl delete -f ./grafana/grafana-sa.yml
    kubectl delete -f ./grafana/grafana-source.yml
    kubectl delete -f ./grafana/grafana-datasources.yml
    kubectl delete -f ./grafana/grafana-admin-secret.yml
    kubectl delete -f ./grafana/grafana-svc.yml
    kubectl delete configmap grafana-config all-grafana-dashboards --namespace=monitoring
    # kubectl create configmap grafana-config --from-file=./grafana/grafana.ini --namespace=monitoring
    # kubectl create configmap all-grafana-dashboards --from-file=./grafana/dashboard --namespace=monitoring
    kubectl delete -f ./grafana/grafana-dp.yml
    sleep 5
    kubectl delete -f ./grafana/grafana-dp.yml
    kubectl delete -f ./service-discovery/kube-controller-manager-svc.yml
    kubectl delete -f ./service-discovery/kube-scheduler-svc.yml
    kubectl delete -f ./prometheus/prometheus-secret.yml
    kubectl delete -f ./prometheus/prometheus-rules.yml
    kubectl delete -f ./prometheus/prometheus-rbac.yml
    kubectl delete -f ./prometheus/prometheus-svc.yml
    # kubectl wait crd/prometheuses.monitoring.coreos.com --for condition=established --timeout=60s
    kubectl delete -f ./prometheus/prometheus-main.yml
    sleep 5
    kubectl delete -f ./prometheus/pv-pvc-hostpath.yaml
    kubectl delete -f ./prometheus/prometheus-main.yml
    kubectl delete -f ./servicemonitor/alertmanager-sm.yml
    kubectl delete -f ./servicemonitor/coredns-sm.yml
    kubectl delete -f ./servicemonitor/kube-apiserver-sm.yml
    kubectl delete -f ./servicemonitor/kube-controller-manager-sm.yml
    kubectl delete -f ./servicemonitor/kube-scheduler-sm.yml
    kubectl delete -f ./servicemonitor/kubelet-sm.yml
    kubectl delete -f ./servicemonitor/kubestate-metrics-sm.yml
    kubectl delete -f ./servicemonitor/node-exporter-sm.yml
    kubectl delete -f ./servicemonitor/prometheus-operator-sm.yml
    kubectl delete -f ./servicemonitor/prometheus-sm.yml
    kubectl delete -f ./servicemonitor/pushgateway-sm.yml
    kubectl delete -f ./prometheus_adapter/metric_rule.yaml
    kubectl delete -f ./prometheus_adapter/prometheus_adapter.yaml
    cd ../


    # 部署gpu的监控
    kubectl delete -f gpu/nvidia-device-plugin.yml
    kubectl delete -f gpu/dcgm-exporter.yaml

    kubectl delete serviceaccount frameworkbarrier --namespace pipeline
    kubectl delete serviceaccount frameworkbarrier --namespace automl
    kubectl delete serviceaccount frameworkbarrier --namespace kubeflow
    kubectl delete clusterrole frameworkbarrier --verb=get,list,watch --resource=frameworks
    kubectl delete clusterrolebinding frameworkbarrier-pipeline --clusterrole=frameworkbarrier  --user=system:serviceaccount:pipeline:frameworkbarrier
    kubectl delete clusterrolebinding frameworkbarrier-automl --clusterrole=frameworkbarrier  --user=system:serviceaccount:automl:frameworkbarrier
    kubectl delete clusterrolebinding frameworkbarrier-kubeflow --clusterrole=frameworkbarrier  --user=system:serviceaccount:kubeflow:frameworkbarrier

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
    kubectl delete -f argo/minio-pv-pvc-hostpath.yaml
    kubectl delete -f argo/pipeline-runner-rolebinding.yaml
    kubectl delete -f argo/install-3.4.3-all.yaml

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

    kubectl delete -f pv-pvc-infra.yaml
    kubectl delete -f pv-pvc-jupyter.yaml
    kubectl delete -f pv-pvc-automl.yaml
    kubectl delete -f pv-pvc-pipeline.yaml
    kubectl delete -f pv-pvc-service.yaml

    kubectl delete -k cube/overlays
    kubectl delete -f sa-rbac.yaml
    sh delete_ns_secret.sh
    kubectl label node $node train- cpu- notebook- service- org- istio- kubeflow- kubeflow-dashboard- mysql- redis- monitoring- logging-
    
    read -p "是否删除所有数据？(yes/no)" answer
    if [[ "$answer" == "yes" ]]; then
        echo "delete all data !!! "
        rm -rf /data/k8s/*
    fi
    echo "ns未删除，可视情况删除ns"
else
  echo "退出脚本！"
  exit 0
fi