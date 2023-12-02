# 离线下载包
```bash
github地址
wget https://github.com/goharbor/harbor/releases/download/v2.3.4/harbor-offline-installer-v2.3.4.tgz
国内网络打包下载地址
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/harbor/harbor-offline-installer-v2.3.4.tgz
解压
tar xf harbor-offline-installer-v2.3.4.tgz -C /usr/local/
cd /usr/local/harbor

cp harbor.yml.tmpl harbor.yml
```

# 修改hostname、harbor登录密码、关闭https。

vim harbor.yml
```bash
hostname: xx.xx.xx.xx
harbor_admin_password: admin
#https:
  # https port for harbor, default is 443
  #  port: 443
  # The path of cert and key files for nginx
  #  certificate: /your/certificate/path
  # private_key: /your/private/key/path
data_volume: /data  #这个路径是宿主机的路径，根据实际情况修改成空间大的地方
```

# 执行安装程序，只安装harbor
```bash
apt install -y docker-compose
./install.sh  (前提条件：docker需要启动)
```

# 除了安装harbor外，还可以安装公正服务 notary 以及漏洞扫描器 trivy，
```bash
./install.sh --with-notary --with-trivy --with-chartmuseum
```

# 用docker-compose查看Harbor容器的运行状态
```bash
docker-compose ps
```

# 配置证书

在 部署好的 Harbor 中添加 HTTPS 证书配置

[harbor镜像仓库-https访问的证书配置](https://zhuanlan.zhihu.com/p/234918875)

[x509: cannot validate certificate for 10.30.0.163 because it doesn't contain any IP SANs](https://blog.csdn.net/min19900718/article/details/87920254)

最后 Docker login $harborIP，就可以 docker pull 拉取服务。
