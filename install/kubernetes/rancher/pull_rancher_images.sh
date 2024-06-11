docker pull rancher/mirrored-cluster-proportional-autoscaler:1.8.3 &
docker pull rancher/mirrored-calico-kube-controllers:v3.19.2 &
docker pull rancher/mirrored-calico-cni:v3.19.2 &
docker pull rancher/mirrored-coredns-coredns:1.8.4 &
docker pull rancher/mirrored-coreos-etcd:v3.4.16-rancher1 &
docker pull rancher/mirrored-metrics-server:v0.5.0 &
docker pull registry:2 &
docker pull rancher/rancher-runtime:v2.6.2 &
docker pull rancher/mirrored-coreos-flannel:v0.14.0 &
docker pull rancher/rke-tools:v0.1.78 &
docker pull rancher/kubectl:v1.20.2 &
docker pull rancher/mirrored-calico-pod2daemon-flexvol:v3.19.2 &
docker pull rancher/mirrored-pause:3.4.1 &
docker pull rancher/rancher:v2.6.2 &
docker pull rancher/shell:v0.1.10 &
docker pull rancher/kube-api-auth:v0.1.5 &
docker pull rancher/rancher-agent:v2.6.2 &
docker pull rancher/hyperkube:v1.21.5-rancher1 &
docker pull rancher/mirrored-kube-rbac-proxy:v0.5.0 &
docker pull rancher/mirrored-calico-node:v3.19.2 &
docker pull rancher/rancher-webhook:v0.2.1 &
docker pull rancher/fleet-agent:v0.3.7

wait
