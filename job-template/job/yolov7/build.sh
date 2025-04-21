#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01 -f Dockerfile  .
docker push ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01 . --push
