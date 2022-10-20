
# 开发者：开发新的AI应用
新建应用目录（可直接复制参考app1应用），在新应用目录下新建init.sh  Dockerfile  app.py文件，

其中 
 - Dockerfile为镜像构建 
 - init.sh位初始化脚本
 - app.py为应用启动(训练/推理/服务)，需要补齐Model类的基础参数
 - 其他自行添加配套内容

镜像调试，基础镜像为conda环境。先使用如下命令启动基础环境进入容器
```bash
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
docker run --name ${aiapp} --privileged -it -v $cube_dir/src:/src -v $PWD:/app -p 8080:8080  ccr.ccs.tencentyun.com/cube-studio/aihub:base bash
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
docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:${aiapp}  .
```

# 用户：部署体验应用
首先需要部署docker
```bash
# 获取当前项目名作为应用名
aiapp=$(basename `pwd`)
cube_dir=($(dirname $(dirname "$PWD")))
docker run --name ${aiapp} --privileged --rm -it -v $cube_dir/src:/src -v $PWD:/app -p 8080:8080 ccr.ccs.tencentyun.com/cube-studio/aihub:${aiapp}
```
