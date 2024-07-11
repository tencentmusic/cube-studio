docker pull rancher/rancher-agent:v2.8.5 &
docker pull rancher/rancher:v2.8.5 &
docker pull rancher/rancher-webhook:v0.4.7 &
docker pull rancher/shell:v0.1.24 &
docker pull rancher/kube-api-auth:v0.2.1 &
docker pull rancher/hyperkube:v1.25.16-rancher2 &
docker pull rancher/calico-cni:v3.26.3-rancher1 &
docker pull rancher/rke-tools:v0.1.96 &
docker pull rancher/mirrored-calico-kube-controllers:v3.26.3 &
docker pull rancher/mirrored-calico-node:v3.26.3 &
docker pull rancher/mirrored-coreos-etcd:v3.5.9 &
docker pull rancher/mirrored-metrics-server:v0.6.2 &
docker pull rancher/mirrored-coredns-coredns:1.9.4 &
docker pull rancher/mirrored-flannelcni-flannel:v0.19.2 &
docker pull rancher/mirrored-cluster-proportional-autoscaler:1.8.6 &
docker pull rancher/mirrored-pause:3.7 &

wait
