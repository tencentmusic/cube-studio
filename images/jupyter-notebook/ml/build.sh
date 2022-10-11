set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

# 构建bigdata镜像
docker build -t  $hubhost/notebook:jupyter-ubuntu-machinelearning -f Dockerfile .
docker push $hubhost/notebook:jupyter-ubuntu-machinelearning
