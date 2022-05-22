![image](https://github.com/tencentmusic/cube-studio/blob/tfra-dev/install/kubernetes/tmeps/docs/tmpps%E6%9E%B6%E6%9E%84%E5%9B%BE.png)

基于tfra v0.3.1: https://github.com/tensorflow/recommenders-addons.git

### 代码结构


### 如何部署

    sh deploy.sh   # 部署一个demo模型

### 如何请求推理服务

跟请求原生tf-serving一样，参考src/client.py


### 如何更改tf和tf serving的版本（仅作参考，具体版本具体分析）

1. 切换子项目的tag或者branch  
2. Dockerfile_tfra_trainer 的 TF_VERSION  
3. 检查Dockerfile_tfra_server 的各种依赖的版本，例如cuda  
4. patch/下的lib_tf，用virturlenv装一个对应的tensorflow，改下tfra_bazelrc前3个参数。  
5. configure.py的_VALID_BAZEL_VERSION加上对应tf版本  



