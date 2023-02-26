#set -ex
#hubhost=ccr.ccs.tencentyun.com/cube-studio
#base_image=ubuntu:18.04
#docker build -t $hubhost/notebook:jupyter-ubuntu-cpu-base --build-arg FROM_IMAGES=$base_image -f Dockerfile-ubuntu-base .
#docker push $hubhost/notebook:jupyter-ubuntu-cpu-base
#
#base_image=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#docker build -t $hubhost/notebook:jupyter-ubuntu-gpu-base --build-arg FROM_IMAGES=$base_image -f Dockerfile-ubuntu-base .
#docker push $hubhost/notebook:jupyter-ubuntu-gpu-base



hubhost=ccr.ccs.tencentyun.com/cube-studio
base_image=ubuntu:22.04
docker build -t $hubhost/notebook:jupyter-ubuntu22.04 --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.8 --build-arg CONDAENV=38 -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu22.04

base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker build -t $hubhost/notebook:jupyter-ubuntu22.04-cuda11.8.0-cudnn8 --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.8 --build-arg CONDAENV=38 -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu22.04-cuda11.8.0-cudnn8

base_image=nvidia/cuda:11.4.0-runtime-ubuntu20.04
docker build -t $hubhost/notebook:jupyter-ubuntu20.04-cuda11.4.0-cudnn8 --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.8 --build-arg CONDAENV=38 -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu20.04-cuda11.4.0-cudnn8


