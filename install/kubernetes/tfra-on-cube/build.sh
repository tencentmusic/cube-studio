your_docker_registry=mirrors.tencent.com/uthermai

# 用于训练过程中，ps、worker等节点
docker build --network=host -t ${your_docker_registry}/tfra-on-cube:trainning -f Dockerfile_tfra_trainning .

# 用于推理服务
docker build --network=host -t ${your_docker_registry}/tfra-on-cube:serving -f Dockerfile_tfra_serving .

#docker build --network=host -t ${your_docker_registry}/tfra-on-cube:serving --build-arg TF_SERVING_VERSION_GIT_COMMIT=2.5.2 -f Dockerfile_tfra_serving .
