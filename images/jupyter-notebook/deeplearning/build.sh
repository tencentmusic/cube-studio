set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

# 构建deeplearning镜像
docker build -t  $hubhost/notebook:jupyter-ubuntu-deeplearning -f Dockerfile .
docker push $hubhost/notebook:jupyter-ubuntu-deeplearning
