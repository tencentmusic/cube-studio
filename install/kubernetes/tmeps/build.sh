
docker build --network=host -t ai.tencentmusic.com/tme-public/tmeps:trainning -f Dockerfile_tfra_trainning . && /
docker build --network=host -t ai.tencentmusic.com/tme-public/tmeps:serving -f Dockerfile_tfra_serving .
docker push ai.tencentmusic.com/tme-public/tmeps:trainning
docker push ai.tencentmusic.com/tme-public/tmeps:serving

#docker build --network=host -t ${your_docker_registry}/tfra-on-cube:serving --build-arg TF_SERVING_VERSION_GIT_COMMIT=2.5.2 -f Dockerfile_tfra_serving .
