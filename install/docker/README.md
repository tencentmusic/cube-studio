
# 本地调试

## deploy mysql

```
docker run -p 3306:3306 --restart always --name mysql -e MYSQL_ROOT_PASSWORD=admin -e MYSQL_ALLOW_EMPTY_PASSWORD=true -v $PWD/docker-add-file/mysqld.cnf:/etc/mysql/mysql.conf.d/mysqld.cnf -d mysql:8.0.32
```
进入mysql，创建kubeflow数据库
```
mysql> CREATE DATABASE IF NOT EXISTS kubeflow DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;

# 然后给admin用户授权，以便其他容器能访问。
mysql> use mysql;
mysql> GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION; 
mysql> flush privileges;
```

## 镜像构建（可忽略）

```
构建基础镜像（包含基础环境）
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:base-python3.9 -f install/docker/Dockerfile-base .

使用基础镜像构建生产镜像
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:2024.03.01 -f install/docker/Dockerfile .

构建frontend镜像
docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard-frontend:2024.03.01 -f install/docker/dockerFrontend/Dockerfile .
```

## 镜像拉取(如果你不参与开发可以直接使用线上镜像)
```
docker pull ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:2024.03.01
docker pull ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard-frontend:2024.03.01
```

## deploy myapp (docker-compose)

注意：前后端代码生产环境均在容器中运行。

后端开发调试：代码通过挂载，在本机ide中修改，在docker-compose中运行调试，debug模式自动热更新。
前端调试：本机ide开发调试，最后编译为静态文件，也可以在docker中编译成静态文件，最后统一打包成docker镜像

#### 本地后端python代码开发

需要安装下面的环境包, python 3.9.16

```bash
pip3 install --upgrade setuptools pip 
pip3 install -r requirements.txt
```
或者使用conda
```bash
conda create -y -n cube-studio python=3.9
source activate cube-studio
conda config --append channels conda-forge
conda install --file requirements.txt
```
本地安装python包，避免本地打开代码时大量包缺失报错


#### 本地后端代码调试

需要在docker-compose运行调试，通过日志进行debug，建议学习下pysnooper包的使用

docker-compose.yaml文件在install/docker目录下，这里提供了mac和linux版本的docker-compose.yaml。

1) debug backend
```
STAGE: 'dev'
docker-compose -f docker-compose.yml  up
```

部署以后，登录首页 会自动创建用户，绑定角色（Gamma和username同名角色）。

可根据自己的需求为角色授权。

#### 前端页面本机开发和构建

前端代码可以在本机上开发调试  也 可以在容器内编译。如果你不是前端开发人员，建议使用容器内编译，这样你的电脑就不需要配置前端环境。

#### 前端代码目录

- `myapp/frontend` 主要前端项目文件
- `myapp/vision` 流程图（AI pipeline）
- `myapp/visionPlus` 流程图（数据ETL pipeline）

项目资源打包：
```
开发环境要求：
node: 16.15.0+
npm: 6.14.8+

包管理（建议使用yarn）：
yarn: npm install yarn -g

```
```sh
# 初始化安装可能会遇到依赖包的版本选择，直接回车默认即可
cd myapp/vision && yarn && yarn build
```
输出路径：`myapp/static/appbuilder`
#### 纯前端开发（本地）

##### 环境准备

- https://nodejs.org/en/download/ 进入nodejs官网，选择下载LTS长期支持版本
- 然后在官网下载安装好LTS版本之后，输入`npm install -g n`安装node版本管理器（ https://www.npmjs.com/package/n ），最后输入`n 16.15.0`将node版本切换至16.x
- https://github.com/nodejs/Release 这里可以找到16.x等往期版本


以主要前端项目`myapp/frontend`为例，到这里前端开发环境已经准备好了

1. `cd myapp/frontend` 进入目录
2. `npm run start` 进入调试模式。需要现在本地启动后端才行
3. `npm run build` 打包编译静态资源
#### 前端配置代理

在前端目录下的`src`里都有一个`setupProxy.js`文件，这个就是用来配置代理转发的，将`target`字段的内容改成已有服务的地址，改完之后重新启动一下前端项目即可生效
#### 前端容器内编译

1) build frontend
```
STAGE: 'build'
docker-compose -f docker-compose.yml  up
```

## Q&A
1） 如果构建镜像过程中因为网络问题失败，可以通过新增pip国内镜像地址来解决。

在cube-studio/install/docker中新建pip.conf，输入以下内容: 
```
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=mirrors.aliyun.com
```
然后在cube-studio/install/docker/Dockerfile-base中增加:

```
COPY install/docker/pip.conf /root/.pip/pip.conf
```

2） 如果前端安装依赖失败，可以通过新增npm国内镜像来解决

npm配置镜像命令
```
npm config set registry https://registry.npmmirror.com
```

yarn配置镜像命令
```
yarn config set registry https://registry.npmmirror.com
```
