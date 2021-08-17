
set -ex
hubhost=ai.tencentmusic.com/tme-public
base_image=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.6 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.6  -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.7  -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .




