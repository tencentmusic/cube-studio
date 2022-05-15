#!/bin/sh
name=$1
namespace=$2
env=$3
while true;
do
#	echo "stern $name --namespace $namespace --kubeconfig /root/.kube/$env-kubeconfig --tail 0 --template '{{.PodName}} {{.Message}}'";
#	timeout 600s stern $name --namespace $namespace --kubeconfig /root/.kube/$env-kubeconfig --tail 0 --template '{{.PodName}} {{.Message}}';
	timeout 600s stern $name --namespace $namespace --kubeconfig /root/.kube/$env-kubeconfig --tail 0 --template '{{.Message}}';
done
