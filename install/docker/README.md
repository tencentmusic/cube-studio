
# 本地调试

## windows 系统下基础环境
1. 安装docker desktop
下载Docker Desktop Installer，安装最新版docker desktop。 windows 10 版本需要启用 Hyper-V ，安装配置流程可参考在线文档https://zhuanlan.zhihu.com/p/441965046

2. 安装 Power Shell
之后的脚本需要在Power Shell 中执行

3. 安装python > 3.9.16

## mac和linux系统基础环境
参考/install/README.md

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
前端调试：大部分功能在本机ide开发调试即可，支持热更新。完整的平台前端调试需要本地ide开发完成后打包编译为静态文件，也可以在docker中编译成静态文件，在docker-compose中运行验证，不同于后端，这种场景下前端不支持热更新。

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

3）windows 系统下执行pip3 install -r requirements.txt 失败
Power Shell 单独安装失败的部分，但要确保安装的版本号与requirements.txt保持一致，安装成功后再重新执行pip3 install -r requirements.txt

4）windows 环境，执行docker-compose报UnicodeDecodeError: 'gbk' codec问题
编码类型问题可参考 https://cloud.tencent.com/developer/article/1530430 修改本地文件 encoding.py  大概位置为：XXX\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip\_internal\utils\encoding.py

5）windows 环境，执行docker-compose，报 Python——/usr/bin/env: ‘python(3)\r’: No such file or directory
原因：在Windows系统中，文本文件的行尾通常以回车符(CR)和换行符(LF)的组合表示（称为CRLF），而在Linux和Unix系统中，行尾仅以换行符(LF)表示。当你在Windows环境下编写或编辑Shell脚本，然后尝试在Linux系统上运行时，就可能会遇到这个问题。
解决方案： vscode 打开项目，全文搜索报错文本关键字，比如/usr/bin/env，打开对应文件，将VSCode 右下角的CRLF 切换为 LF 保存对应文件

6）windows 环境，打包visionPlus 编译过程报错
/myapp/visionPlus/.eslintrc 文件中 注释行："linebreak-style": ["error", "unix"]，取消注释行："linebreak-style": ["error", "windows"]

