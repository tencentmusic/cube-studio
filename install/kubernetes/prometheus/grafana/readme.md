kubectl delete -n monitoring configmaps grafana-config
kubectl create configmap grafana-config --from-file=./grafana.ini --namespace=monitoring 

kubectl delete -n monitoring configmaps grafana-defaults-config
kubectl create configmap grafana-defaults-config --from-file=./defaults.ini --namespace=monitoring 