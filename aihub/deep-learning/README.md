
#  [视频教程 快速入门](https://www.bilibili.com/video/BV1X84y1y7xy/?vd_source=bddb004da42430029e7bd52d0bdd0fe7)

# 开发者：开发新的AI应用
新建应用目录（可直接复制参考app1应用），在新应用目录下新建init.sh  Dockerfile  app.py文件，

其中 
 - Dockerfile为镜像构建 
 - init.sh位初始化脚本
 - app.py为应用启动(训练/推理/服务)，需要补齐Model类的基础参数
 - 其他自行添加配套内容

镜像调试，基础镜像为conda环境。先使用如下命令启动基础环境进入容器

ccr.ccs.tencentyun.com/cube-studio/aihub:base 无python环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9 为conda，python3.9环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.8 为conda，python3.8环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.6 为conda，python3.6环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4 无python环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.6  为conda，python3.6环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.8  为conda，python3.8环境
ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.9  为conda，python3.9环境

```bash
# 进入模型应用
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
chmod +x $cube_dir/src/docker/entrypoint.sh
sudo docker run --name ${aiapp} --privileged -it -e APPNAME=$aiapp -v $cube_dir/src:/src -v $PWD:/app -p 80:80 -p 8080:8080 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9 bash 

```
如果需要使用gpu调试
```bash
sudo docker run --name ${aiapp} --privileged -it --gpu=0  -e APPNAME=$aiapp -e NVIDIA_VISIBLE_DEVICES=all -v $cube_dir/src:/src -v $PWD:/app -p 80:80 -p 8080:8080 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9 bash 
```
补全init.sh环境脚本。
```bash
# init.sh 脚本会被复制到容器/根目录下，环境文件不要放置在容器/app/目录下，不然会被加载到git
cp init.sh /init.sh && bash /init.sh
```
补齐app.py，运行调试
```bash
python app.py
```
生成aiapp的镜像
```bash
aiapp=$(basename `pwd`)
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:${aiapp}  .
```

# 用户：部署体验应用
首先需要部署docker
```bash
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
chmod +x $cube_dir/src/docker/entrypoint.sh
sudo docker run --name ${aiapp} --rm -it -e APPNAME=$aiapp -v $cube_dir/src:/src -v $PWD:/app -p 80:80 -p 8080:8080 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/aihub:${aiapp} python app.py 

```
如果是gpu服务
```bash
sudo docker run --name ${aiapp} --privileged --rm -it  -e APPNAME=$aiapp -e NVIDIA_VISIBLE_DEVICES=all -v $cube_dir/src:/src -v $PWD:/app -p 80:80 -p 8080:8080 --entrypoint='/src/docker/entrypoint.sh' ccr.ccs.tencentyun.com/cube-studio/aihub:${aiapp} python app.py 

```
