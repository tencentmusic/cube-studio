通过nerdctl离线安装Harbor仓库。

# 下载离线安装包
wget https://cube-studio.oss-cn-hangzhou.aliyuncs.com/harbor/harbor-offline-installer-v2.3.4.tgz

# 解压并拷贝至/usr/local/目录下

tar xf harbor-offline-installer-v2.3.4.tgz -C /usr/local/

# 进入/usr/local/harbor目录
cd /usr/local/harbor
 
# 复制harbor配置文件
cp harbor.yml.tmpl harbor.yml

# 注释配置文件中的https,修改http端口，并设置hostname为主机ip
vi harbor.yml
默认密码 Harbor12345
# 修改install.sh，common.sh prepare 文件中的docker-compose为nerdctl compose，docker为nerdctl
   
sed -i 's/docker-compose/nerdctl compose/g' install.sh
sed -i 's/docker/nerdctl/g' install.sh

sed -i 's/docker-compose/nerdctl compose/g' common.sh
sed -i 's/docker/nerdctl/g' common.sh

sed -i 's/docker-compose/nerdctl compose/g' prepare
sed -i 's/docker/nerdctl/g' prepare

# 执行安装
1、vi install.sh中去除check_nerdctl check_nerdctlcompose等命令
2、bash ./install.sh


# 卸载
cd /usr/local/harbor
nerdctl compose down
rm -rf `find / -name harbor`
# 将运行的容器全部停止
nerdctl stop `nerdctl ps -q`
# 将容器全部删除
nerdctl rm `nerdctl ps -qa`
# 将镜像全部删除
nerdctl rmi `nerdctl images -q`
