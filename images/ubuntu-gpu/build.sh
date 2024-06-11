set -ex
TARGETARCH=amd64
hubhost=ccr.ccs.tencentyun.com/cube-studio

base_image=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

docker build --network=host -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build --network=host -t $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-$TARGETARCH --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH --build-arg PYTHON_VERSION=python3.9 -f cuda/python/Dockerfile .

docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-$TARGETARCH
docker push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-$TARGETARCH

# docker manifest create $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9 $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-amd64 $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9-arm64 && docker manifest push $hubhost/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9

base_image=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

docker build --network=host -t $hubhost/ubuntu-gpu:cuda12.1.0-cudnn8-$TARGETARCH --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build --network=host -t $hubhost/ubuntu-gpu:cuda12.1.0-cudnn8-python3.9-$TARGETARCH --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda12.2.2-cudnn8-$TARGETARCH --build-arg PYTHON_VERSION=python3.9 -f cuda/python/Dockerfile .

docker push $hubhost/ubuntu-gpu:cuda12.1.0-cudnn8-$TARGETARCH
docker push $hubhost/ubuntu-gpu:cuda12.1.0-cudnn8-python3.9-$TARGETARCH


