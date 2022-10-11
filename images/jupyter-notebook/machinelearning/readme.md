
## 一、基于Dockerfile实现

### 1.1 添加测试示例
在example下面添加内部常用机器学习算法示例

平台会自动将example软链到用户个人目录下

### 1.2 通过Dockerfile构建镜像
```bash
docker build -t  $hubhost/notebook:jupyter-ubuntu-machinelearning -f Dockerfile .
docker push $hubhost/notebook:jupyter-ubuntu-machinelearning
```

### 1.3 上线自己的notebook镜像到cube-studio

config.py中 NOTEBOOK_IMAGES 变量为notebook可选镜像，更新此变量即可。

