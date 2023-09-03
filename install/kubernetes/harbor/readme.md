wget https://github.com/goharbor/harbor/releases/download/v2.8.1/harbor-offline-installer-v2.8.1.tgz

tar xf harbor-offline-installer-v2.3.4.tgz -C /usr/local/
cd /usr/local/harbor

cp harbor.yml.tmpl harbor.yml

#修改hostname、harbor登录密码、关闭https。

vim harbor.yml
hostname: www.myharbor.com
harbor_admin_password: harbor12345
#https:
  # https port for harbor, default is 443
  #  port: 443
  # The path of cert and key files for nginx
  #  certificate: /your/certificate/path
  # private_key: /your/private/key/path
data_volume: /data  #这个路径是宿主机的路径，根据实际情况修改成空间大的地方

#执行安装程序，只安装harbor
./install.sh  (前提条件：docker需要启动)

# 除了安装harbor外，还安装公正服务 notary 以及漏洞扫描器 trivy，
./install.sh --with-notary --with-trivy --with-chartmuseum

用docker-compose查看Harbor容器的运行状态