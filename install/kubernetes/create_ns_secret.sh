
for namespace in 'infra' 'kubeflow' 'istio-system' 'knative-serving' 'pipeline' 'katib' 'jupyter' 'kfserving' 'service' 'cert-manager' 'monitoring' 'logging' 'kube-system'
do
    kubectl create ns $namespace
    kubectl delete secret docker-registry hubsecret -n $namespace
    kubectl create secret docker-registry hubsecret --docker-username=xxx --docker-password=xxxx -n $namespace
    kubectl label ns $namespace istio-injection=disabled --overwrite
#    kubectl label namespace $namespace istio-inhection=enabled --overwrite
done

kubectl label ns katib katib-metricscollector-injection=enabled --overwrite



