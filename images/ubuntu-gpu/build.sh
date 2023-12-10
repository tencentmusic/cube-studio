set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

base_image=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
TARGETARCH=amd64
docker build -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.7-$TARGETARCH --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.8-$TARGETARCH --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-$TARGETARCH --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg PYTHON_VERSION=python3.9 -f cuda/python/Dockerfile .

docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH
docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.7-$TARGETARCH
docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.8-$TARGETARCH
docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-$TARGETARCH
