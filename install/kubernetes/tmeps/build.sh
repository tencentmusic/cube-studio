
docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/tmeps:trainning -f Dockerfile_tfra_trainning . && /
docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/tmeps:serving -f Dockerfile_tfra_serving .
docker push ccr.ccs.tencentyun.com/cube-studio/tmeps:trainning
docker push ccr.ccs.tencentyun.com/cube-studio/tmeps:serving

#docker build --network=host -t ${your_docker_registry}/tfra-on-cube:serving --build-arg TF_SERVING_VERSION_GIT_COMMIT=2.5.2 -f Dockerfile_tfra_serving .
