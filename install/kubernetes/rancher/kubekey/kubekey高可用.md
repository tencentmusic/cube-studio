# 可以正常添加节点 

需要注意address
```bash
***  主要address    ***
  controlPlaneEndpoint:
    ## Internal loadbalancer for apiservers
    # internalLoadbalancer: haproxy
    domain: lb.kubesphere.local
*** address: "172.21.106.247" ***
```

# 节点可正常连接时删除master节点

`./kk delete node <nodeName> -f config-cluster.yaml`

删除从cluster-config.yaml中再删除节点信息，加入新的master节点信息

# 节点不可以正常连接时删除master节点

```bash
kubectl drain nodename --ignore-saemonsets=true --grace-period=200

kubectl delete node nodename 

# 从etcd集群列出成员
exist_host_ip1=172.17.0.3
exist_host_ip2=172.17.0.252
ETCDCTL_API=3 etcdctl \
--endpoints=https://$exist_host_ip1:2379 \
--cacert=/etc/ssl/etcd/ssl/ca.pem \
--cert=/etc/ssl/etcd/ssl/admin-$exist_host_ip1.pem \
--key=/etc/ssl/etcd/ssl/admin-$exist_host_ip1-key.pem \
member list
  
# 获取被删除的节点在etcd中的名称
# 从etcd集群移除（替换最后面的节点名称）
ETCDCTL_API=3 etcdctl \
--endpoints=https://$exist_host_ip1:2379,https://$exist_host_ip2:2379 \
--cacert=/etc/ssl/etcd/ssl/ca.pem \
--cert=/etc/ssl/etcd/ssl/admin-$exist_host_ip1.pem \
--key=/etc/ssl/etcd/ssl/admin-$exist_host_ip1-key.pem \
member remove b66918af2a6cdd2c  

```
删除旧的master，再加入新master节点