set -ex
TARGETARCH=amd64

hubhost=ccr.ccs.tencentyun.com/cube-studio

base_image=ubuntu:20.04
docker build --network=host -t $hubhost/notebook:jupyter-ubuntu-cpu-conda-$TARGETARCH --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.9 --build-arg CONDAENV=python39 --build-arg TARGETARCH=$TARGETARCH -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu-cpu-conda-$TARGETARCH

base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
docker build --network=host -t $hubhost/notebook:jupyter-ubuntu-gpu-conda-$TARGETARCH --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.9 --build-arg CONDAENV=python39 --build-arg TARGETARCH=$TARGETARCH --build-arg CUDAVERSION=cu118 -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu-gpu-conda-$TARGETARCH

base_image=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
docker build --network=host -t $hubhost/notebook:jupyter-ubuntu-gpu-conda-12.1.0-cudnn8-$TARGETARCH --build-arg FROM_IMAGES=$base_image --build-arg PYTHONVERSION=3.9 --build-arg CONDAENV=python39 --build-arg TARGETARCH=$TARGETARCH --build-arg CUDAVERSION=cu121 -f Dockerfile-ubuntu-conda .
docker push $hubhost/notebook:jupyter-ubuntu-gpu-conda-12.1.0-cudnn8-$TARGETARCH


