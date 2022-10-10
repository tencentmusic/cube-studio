# 创建sa-rbac
```bash
kubectl create serviceaccount frameworkcontroller --namespace service
kubectl create serviceaccount frameworkcontroller --namespace pipeline
kubectl create serviceaccount frameworkcontroller --namespace automl
kubectl create serviceaccount frameworkcontroller --namespace kubeflow
kubectl create clusterrolebinding frameworkcontroller-service --clusterrole=cluster-admin --user=system:serviceaccount:service:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-pipeline --clusterrole=cluster-admin --user=system:serviceaccount:pipeline:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-automl --clusterrole=cluster-admin --user=system:serviceaccount:automl:frameworkcontroller
kubectl create clusterrolebinding frameworkcontroller-kubeflow --clusterrole=cluster-admin --user=system:serviceaccount:kubeflow:frameworkcontroller

```


# 默认配置的frameworkcontroller
```bash
kubectl create -f frameworkcontroller-with-default-config.yaml

kubectl create serviceaccount frameworkbarrier --namespace service
kubectl create serviceaccount frameworkbarrier --namespace pipeline
kubectl create serviceaccount frameworkbarrier --namespace automl
kubectl create serviceaccount frameworkbarrier --namespace kubeflow
kubectl create clusterrole frameworkbarrier --verb=get,list,watch --resource=frameworks
kubectl create clusterrolebinding frameworkbarrier-service --clusterrole=frameworkbarrier  --user=system:serviceaccount:service:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-pipeline --clusterrole=frameworkbarrier  --user=system:serviceaccount:pipeline:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-automl --clusterrole=frameworkbarrier  --user=system:serviceaccount:automl:frameworkbarrier
kubectl create clusterrolebinding frameworkbarrier-kubeflow --clusterrole=frameworkbarrier  --user=system:serviceaccount:kubeflow:frameworkbarrier

```


# 也可以自定义配置的frameworkcontroller
kubectl create -f frameworkcontroller-customized-config.yaml
kubectl create -f frameworkcontroller-with-customized-config.yaml