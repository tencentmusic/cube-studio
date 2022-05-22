set -ex

kubectl --kubeconfig /mnt/uthermai/job-template/job/pkgs/config/tke-kubeconfig apply -f resources/ns.yaml
sleep 2

kubectl --kubeconfig /mnt/uthermai/job-template/job/pkgs/config/tke-kubeconfig apply -f conf/
kubectl --kubeconfig /mnt/uthermai/job-template/job/pkgs/config/tke-kubeconfig apply -f resources/redis_standlone.yaml
#kubectl --kubeconfig /mnt/uthermai/job-template/job/pkgs/config/tke-kubeconfig wait -f resources/redis_standlone.yaml --for condition=available
sleep 2

if [ -z $(which kustomize) ];then
        echo  "kustomize 未安装"
	curl -Lo ./kustomize https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
	chmod 777 ./kustomize
	mv ./kustomize /usr/local/bin/
else
        echo "kustomize 已安装"
fi

kustomize build resources | kubectl delete -f -
sleep 2

#kustomize build resources | kubectl apply -f -

