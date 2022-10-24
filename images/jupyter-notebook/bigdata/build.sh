set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

# 构建bigdata镜像
docker build -t  $hubhost/notebook:jupyter-ubuntu-bigdata -f Dockerfile .
#docker build --build-arg APACHE_MIRROR=https://mirrors.aliyun.com/apache --build-arg  PIPI_MIRROR_ENABLE=true --build-arg  UBUNTU_MIRROR_ENABLE=true -t  $hubhost/notebook:jupyter-ubuntu-bigdata -f Dockerfile .
docker push $hubhost/notebook:jupyter-ubuntu-bigdata
