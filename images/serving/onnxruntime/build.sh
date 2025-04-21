set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime/dockerfiles
git submodule update --init

docker build --network=host -t $hubhost/onnxruntime:cpu -f Dockerfile.source ..
docker push $hubhost/onnxruntime:cpu

docker build --network=host -t $hubhost/onnxruntime:cuda -f Dockerfile.cuda ..
docker push $hubhost/onnxruntime:cuda

docker build --network=host -t $hubhost/onnxruntime:tensorrt -f Dockerfile.tensorrt ..
docker push $hubhost/onnxruntime:tensorrt

# 地址：https://github.com/microsoft/onnxruntime/tree/master/dockerfiles

