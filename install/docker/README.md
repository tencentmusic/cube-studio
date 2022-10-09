
# 本地调试

## deploy mysql

```
linux
docker run --network host --restart always --name mysql -e MYSQL_ROOT_PASSWORD=admin -d mysql:5.7
mac
docker run -p 3306:3306 --restart always --name mysql -e MYSQL_ROOT_PASSWORD=admin -d mysql:5.7

```
进入mysql，创建kubeflow数据库
```
mysql> CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;

# 然后给admin用户授权，以便其他容器能访问。
mysql> use mysql;
mysql> GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'admin' WITH GRANT OPTION; 
mysql> flush privileges;
```

## 镜像构建（可忽略）

```
构建基础镜像（包含基础环境）
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:base -f install/docker/Dockerfile-base .

使用基础镜像构建生产镜像
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:2022.09.01 -f install/docker/Dockerfile .

构建frontend镜像
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:frontend-2022.09.01 -f install/docker/dockerFrontend/Dockerfile .
```

## 镜像拉取(如果你不参与开发可以直接使用线上镜像)
```
docker pull ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:2022.09.01
docker pull ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:frontend-2022.09.01
```

## deploy myapp (docker-compose)

本地开发使用

docker-compose.yaml文件在install/docker目录下，这里提供了mac和linux版本的docker-compose.yaml。

可自行修改

image：刚才构建的镜像

MYSQL_SERVICE：mysql的地址

1) build frontend
```
STAGE: 'build'
docker-compose -f docker-compose.yml  up
```
2) debug backend
```
STAGE: 'dev'
docker-compose -f docker-compose.yml  up
```
3) Production
```
STAGE: 'prod'
docker-compose -f docker-compose.yml  up
```

部署以后，登录首页 会自动创建用户，绑定角色（Gamma和username同名角色）。

可根据自己的需求为角色授权。


## 可视化页面

项目资源打包：
```
开发环境要求：
node: 14.15.0+
npm: 6.14.8+

包管理（建议使用yarn）：
yarn: npm install yarn -g
```
```sh
# 初始化安装可能会遇到依赖包的版本选择，直接回车默认即可
cd myapp/vision && yarn && yarn build
```

输出路径：`/myapp/static/appbuilder`

## Q&A
如果构建镜像过程中因为网络问题失败，可以通过新增pip国内镜像地址来解决。

在cube-studio/install/docker中新建pip.conf，输入以下内容: 
```
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://pypi.tuna.tsinghua.edu.cn/simple/

[install]
trusted-host=mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn
```
然后在cube-studio/install/docker/Dockerfile-base中增加:

```
COPY install/docker/pip.conf /root/.pip/pip.conf
```




