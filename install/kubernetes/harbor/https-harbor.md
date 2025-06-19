参考https://blog.csdn.net/networken/article/details/107502461

# 生成配置https证书

这里以registry.harbor.com域名为例进行演示

```bash
mkdir /root/ssl
cd /root/ssl

# 生成CA证书私钥
openssl genrsa -out ca.key 4096

# 生成ca证书
openssl req -x509 -new -nodes -sha512 -days 3650 \
 -subj "/C=CN/ST=Beijing/L=Beijing/O=example/OU=Personal/CN=registry.harbor.com" \
 -key ca.key \
 -out ca.crt


# 生成服务器证书
# 生成私钥
openssl genrsa -out registry.harbor.com.key 4096

# 生成证书签名请求（CSR）
openssl req -sha512 -new \
    -subj "/C=CN/ST=Beijing/L=Beijing/O=example/OU=Personal/CN=registry.harbor.com" \
    -key registry.harbor.com.key \
    -out registry.harbor.com.csr


#生成一个x509 v3扩展文件
cat > v3.ext <<-EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1=registry.harbor.com
DNS.2=registry.harbor
DNS.3=harbor
EOF


# 使用该v3.ext文件为您的Harbor主机生成证书
openssl x509 -req -sha512 -days 3650 \
    -extfile v3.ext \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -in registry.harbor.com.csr \
    -out registry.harbor.com.crt

#生成后ca.crt，yourdomain.com.crt和yourdomain.com.key文件，必须将它们提供给harbor和docker，和重新配置harbor使用它们。

# 转换registry.harbor.com.crt为registry.harbor.com.cert，供Docker使用。
openssl x509 -inform PEM -in registry.harbor.com.crt -out registry.harbor.com.cert

# 将服务器证书，密钥和CA文件复制到Harbor主机上的Docker证书文件夹中

mkdir -p /etc/docker/certs.d/registry.harbor.com/
cp registry.harbor.com.cert /etc/docker/certs.d/registry.harbor.com/
cp registry.harbor.com.key /etc/docker/certs.d/registry.harbor.com/
cp ca.crt /etc/docker/certs.d/registry.harbor.com/

systemctl restart docker

```

# 离线下载包

```bash
github地址
wget https://githubfast.com/goharbor/harbor/releases/download/v2.11.1/harbor-offline-installer-v2.11.1.tgz
# wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/install/harbor-offline-installer-v2.11.1.tgz
解压
tar xf harbor-offline-installer-v2.11.1.tgz -C /usr/local/
cd /usr/local/harbor

cp harbor.yml.tmpl harbor.yml
```

# 修改hostname、harbor登录密码、配置https证书位置。

vim harbor.yml

```bash

hostname: registry.harbor.com

# http related config
#http:
  # port for http, default is 80. If https enabled, this port will redirect to https port
  #port: 80

# https related config
https:
  # https port for harbor, default is 443
  port: 443
  # The path of cert and key files for nginx
  certificate: /root/ssl/registry.harbor.com.crt
  private_key: /root/ssl/registry.harbor.com.key
  # enable strong ssl ciphers (default: false)
  # strong_ssl_ciphers: false

data_volume: /data  #这个路径是宿主机的路径，根据实际情况修改成空间大的地方
# external_url: http://xx.xx.xx.xx:88   # 如果有公网地址，这里填写
```

admin默认密码 Harbor12345

# 执行harbor部署
```bash
./install.sh

# 每次修改了配置文件都要删除之前的配置，重新安装
cd /usr/local/harbor
rm -rf ./common/config
./install.sh
```

# 配置harbor开机自启

```bash
cat > /usr/lib/systemd/system/harbor.service << 'EOF'
[Unit]
Description=Harbor
After=docker.service systemd-networkd.service systemd-resolved.service
Requires=docker.service
Documentation=http://github.com/vmware/harbor

[Service]
Type=simple
Restart=on-failure
RestartSec=5
Environment=harbor_install_path=/data/packages
ExecStart=/usr/local/bin/docker-compose -f ${harbor_install_path}/harbor/docker-compose.yml up
ExecStop=/usr/local/bin/docker-compose -f ${harbor_install_path}/harbor/docker-compose.yml down

[Install]
WantedBy=multi-user.target
EOF
```

