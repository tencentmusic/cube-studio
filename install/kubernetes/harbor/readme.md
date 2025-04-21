# 离线下载包
```bash
github地址
wget https://githubfast.com/goharbor/harbor/releases/download/v2.11.1/harbor-offline-installer-v2.11.1.tgz
#wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/harbor-offline-installer-v2.11.1.tgz
解压
tar xf harbor-offline-installer-v2.11.1.tgz -C /usr/local/
cd /usr/local/harbor

cp harbor.yml.tmpl harbor.yml
```

# 修改hostname、harbor登录密码、关闭https。

`vim harbor.yml`

修改内容如下，主要涉及 hostname改为ip，http端口可以换掉，https块去掉
```bash
hostname: xx.xx.xx.xx
harbor_admin_password: admin
http:
  # port for http, default is 80. If https enabled, this port will redirect to https port
  port: 88
#https:
  # https port for harbor, default is 443
  #  port: 443
  # The path of cert and key files for nginx
  #  certificate: /your/certificate/path
  # private_key: /your/private/key/path
data_volume: /data  #这个路径是宿主机的路径，根据实际情况修改成空间大的地方
# external_url: http://xx.xx.xx.xx:88   # 如果有公网地址，这里填写
```
admin默认密码 Harbor12345

# 执行安装程序，只安装harbor

安装harbor前需要先安装docker和docker-compose，并且启动docker和docker-compose。

安装docker 参考 install/kubernetes/rancher/install_docker.md

```bash
cd /usr/local/harbor
# 每次修改了配置文件都要删除之前的配置，重新安装
rm -rf ./common/config
./install.sh
```

# 用docker-compose查看Harbor容器的运行状态
```bash
docker-compose ps
```

# 使用http仓库服务

`vi /etc/docker/daemon.json`
添加配置
```bash
{
    "insecure-registries":["xx.xx.xx.xx:88"]
}
```
重启生效
```bash
systemctl stop docker
systemctl daemon-reload
systemctl start docker
```

# 配置证书使用https仓库服务(可忽略)

在 部署好的 Harbor 中添加 HTTPS 证书配置

[harbor镜像仓库-https访问的证书配置](https://zhuanlan.zhihu.com/p/234918875)

[x509: cannot validate certificate for 10.30.0.163 because it doesn't contain any IP SANs](https://blog.csdn.net/min19900718/article/details/87920254)

最后 Docker login $harborIP，就可以 docker pull 拉取服务。

