#!/usr/bin/env bash 
export CRI_CONFIG_FILE=/var/lib/rancher/rke2/agent/etc/crictl.yaml
cp /var/lib/rancher/rke2/bin/crictl /usr/local/bin/
crictl ps

crictl rm -f $(crictl ps -qa)
rm -rf /etc/kubernetes \
       /opt/cni \
       /run/secrets/kubernetes.io \
       /run/calico \
       /run/flannel \
       /var/lib/calico \
       /var/lib/etcd \
       /var/lib/cni \
       /var/lib/kubelet \
       /var/lib/rancher/rke/log \
       /var/log/containers \
       /var/log/pods \
       /var/run/calico \
       /var/etcd

for mount in $(mount | grep tmpfs | grep '/var/lib/kubelet' | awk '{ print $3 }') /var/lib/kubelet /var/lib/rancher; do umount $mount; done
for m in $(sudo tac /proc/mounts | sudo awk '{print $2}'|sudo grep /var/lib/kubelet);do
 sudo umount $m||true
done

for m in $(sudo tac /proc/mounts | sudo awk '{print $2}'|sudo grep /var/lib/rancher);do
 sudo umount $m||true
done

for m in $(sudo tac /proc/mounts | sudo awk '{print $2}'|sudo grep /run/k3s/);do
 sudo umount $m||true
done

crictl volume rm $(crictl volume ls -q)
crictl ps -a
crictl volume ls

pkill -f containerd-shim-runc-v2

sudo rm -rf /var/lib/rancher/
sudo rm -rf /etc/rancher/
sudo rm -rf /run/kubernetes/
sudo rm -rf /var/lib/kubelet/
sudo rm -rf /run/k3s/

rm -f /var/lib/containerd/io.containerd.metadata.v1.bolt/meta.db

sudo systemctl restart containerd