set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

base_image=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
docker build -t $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.0.3-cudnn8 --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda11.0.3-cudnn8 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker push $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8
docker push $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8-python3.7
docker push $hubhost/ubuntu-gpu:cuda11.0.3-cudnn8-python3.8


base_image=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
docker build -t $hubhost/ubuntu-gpu:cuda10.2-cudnn7 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.2-cudnn7-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.2-cudnn7 --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.2-cudnn7-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.2-cudnn7 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker push $hubhost/ubuntu-gpu:cuda10.2-cudnn7
docker push $hubhost/ubuntu-gpu:cuda10.2-cudnn7-python3.7
docker push $hubhost/ubuntu-gpu:cuda10.2-cudnn7-python3.8


base_image=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.6 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.6 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.1-cudnn7 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker push $hubhost/ubuntu-gpu:cuda10.1-cudnn7
docker push $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.6
docker push $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.7
docker push $hubhost/ubuntu-gpu:cuda10.1-cudnn7-python3.8


base_image=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
docker build -t $hubhost/ubuntu-gpu:cuda10.0-cudnn7 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.6 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.0-cudnn7 --build-arg PYTHON_VERSION=python3.6 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.0-cudnn7 --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda10.0-cudnn7 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker push $hubhost/ubuntu-gpu:cuda10.0-cudnn7
docker push $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.6
docker push $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.7
docker push $hubhost/ubuntu-gpu:cuda10.0-cudnn7-python3.8

base_image=nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
docker build -t $hubhost/ubuntu-gpu:cuda9.1-cudnn7 --build-arg FROM_IMAGES=$base_image -f cuda/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.6 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda9.1-cudnn7 --build-arg PYTHON_VERSION=python3.6 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.7 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda9.1-cudnn7 --build-arg PYTHON_VERSION=python3.7 -f cuda/python/Dockerfile .
docker build -t $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.8 --build-arg FROM_IMAGES=$hubhost/ubuntu-gpu:cuda9.1-cudnn7 --build-arg PYTHON_VERSION=python3.8 -f cuda/python/Dockerfile .
docker push $hubhost/ubuntu-gpu:cuda9.1-cudnn7
docker push $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.6
docker push $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.7
docker push $hubhost/ubuntu-gpu:cuda9.1-cudnn7-python3.8


