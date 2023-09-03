
### 在线安装(所有节点)
ubuntu
```shell
apt update
apt install -y nfs-kernel-server
apt install -y nfs-common
```
centos
```shell
yum install -y nfs-utils rpcbind

```
### centos使用rpm包方式安装(所有节点)

```shell
wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/deploy/nfs/nfsrpm.tar.gz
tar -zxvf nfsrpm.tar.gz
cd nfs
rpm -ivh *.rpm --force --nodeps
```

### nfs server配置

```shell

# 修改配置文件，增加下面这一行数据，指定的ip地址为客户端的地址
# /data/nfs/ 代表nfs server本地存储目录
mkdir - p /data/nfs
echo "/data/nfs/ *(rw,no_root_squash,async)" >> /etc/exports

# 加载配置文件
exportfs -arv
systemctl enable rpcbind.service 
systemctl enable nfs-server.service
systemctl start rpcbind.service
systemctl start nfs-server.service

#验证
[root@nfs ~]# showmount -e localhost
Export list for localhost:
/data/nfs   *
```

### nfs client配置

```shell
#查看nfs server 信息

[root@node02 ~]# showmount -e 172.16.101.13
Export list for 172.16.101.13:
/data/nfs      *

# 系统层面添加挂载添加一行，重启自动添加
mkdir -p /data/cube-studio/nfs
echo "172.16.101.13:/data/nfs  /data/cube-studio/nfs   nfs   defaults  0  0" >> /etc/fstab

mount -a 

# 或者使用命令行,如果是一台机器记得不要是同一个地址
mount -t nfs 172.16.101.13:/data/nfs /data/cube-studio/nfs

#验证
df -h |grep nfs

# 软链到cube studio的目录
mkdir -p /data/cube-studio/nfs/k8s
ln -s /data/cube-studio/nfs/k8s /data/

#输出 表示挂载成功
[root@node02 ~]# df -h /data/cube-studio/nfs/
Filesystem                Size  Used Avail Use% Mounted on
172.16.101.13:/data/nfs  3.5T  626M  3.5T   1% /data/cube-studio/nfs

```

