docker login
docker pull rancher/rancher-agent:v2.5.2 &
docker pull rancher/hyperkube:v1.18.20-rancher1 &
docker pull rancher/mirrored-metrics-server:v0.3.6 &
docker pull rancher/mirrored-calico-cni:v3.13.4 &
docker pull rancher/fleet-agent:v0.3.1 &
docker pull registry:2 &
docker pull rancher/mirrored-nginx-ingress-controller-defaultbackend:1.5-rancher1 &
docker pull rancher/mirrored-coredns-coredns:1.6.9 &
docker pull rancher/rke-tools:v0.1.75 &
docker pull rancher/mirrored-calico-pod2daemon-flexvol:v3.13.4 &
docker pull rancher/mirrored-pause:3.1 &
docker pull rancher/mirrored-coreos-etcd:v3.4.15-rancher1 &
docker pull rancher/nginx-ingress-controller:nginx-0.35.0-rancher2 &
docker pull rancher/mirrored-calico-node:v3.13.4 &
docker pull rancher/rke-tools:v0.1.65 &
docker pull busybox &
docker pull rancher/rancher:v2.5.2 &
docker pull rancher/mirrored-coreos-flannel:v0.15.1 &
docker pull rancher/mirrored-cluster-proportional-autoscaler:1.7.1 &
docker pull rancher/kube-api-auth:v0.1.4 &

wait
