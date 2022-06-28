set -ex
hubhost=ccr.ccs.tencentyun.com/cube-studio

git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime/dockerfiles
docker build -t $hubhost/onnxruntime:cpu -f Dockerfile.source ..
docker push $hubhost/onnxruntime:cpu

docker build -t $hubhost/onnxruntime:cuda -f Dockerfile.cuda ..
docker push $hubhost/onnxruntime:cuda

docker build -t $hubhost/onnxruntime:tensorrt -f Dockerfile.tensorrt ..
docker push $hubhost/onnxruntime:tensorrt

docker build -t $hubhost/onnxruntime:tensorrt -f Dockerfile.tensorrt ..
docker push $hubhost/onnxruntime:tensorrt

# 地址：https://github.com/microsoft/onnxruntime/tree/master/dockerfiles
