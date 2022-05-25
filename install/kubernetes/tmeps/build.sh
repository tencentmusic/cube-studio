
your_docker_registry=ai.tencentmusic.com/tme-public

docker build --network=host -t ${your_docker_registry}/tmeps:trainning -f Dockerfile_tfra_trainning . && /
docker build --network=host -t ${your_docker_registry}/tmeps:serving -f Dockerfile_tfra_serving .


#docker build --network=host -t ${your_docker_registry}/tfra-on-cube:serving --build-arg TF_SERVING_VERSION_GIT_COMMIT=2.5.2 -f Dockerfile_tfra_serving .
