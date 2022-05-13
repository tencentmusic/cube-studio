![image](https://github.com/tencentmusic/cube-studio/blob/tfra-dev/install/kubernetes/tmeps/docs/tmpps%E6%9E%B6%E6%9E%84%E5%9B%BE.png)

### 代码结构

serving: tf serving2.5.2 原生代码  
recommenders-addons: tfra0.3.1 原生代码  
patch: 对tf serving镜像和recommenders-addons镜像的各种补丁  
deploy: 部署到k8s的yaml文件  
src: tf模型、数据类  

### 如何部署训练

1. 打包镜像。
    
    ./build.sh    
    
2. 部署redis。已经有redis就不用部署了。  

    kubectl apply -f deploy/redis.yaml
    
3. 调整模型和数据
    模型：model_fn_builder.py
    数据输入：input_fn_builder.py（支持实时消费的方式，详见demo的实现）

3. 部署训练集群。在deploy/kustomization.yaml调整redis host、集群规模等。

    kubectl apply -k deploy/kustomization.yaml

### 如何部署推理服务


### 如何更改tf和tf serving的版本

1. 切换子项目的tag或者branch  
2. Dockerfile_tfra_trainer 的 TF_VERSION  
3. 检查Dockerfile_tfra_server 的各种依赖的版本，例如cuda  
4. patch/下的lib_tf，用virturlenv装一个对应的tensorflow，改下tfra_bazelrc前3个参数。  
5. configure.py的_VALID_BAZEL_VERSION加上对应tf版本  
