#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/yolov8:20250801 -f Dockerfile  .
docker push ccr.ccs.tencentyun.com/cube-studio/yolov8:20250801

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/yolov8:20250801 -f Dockerfile . --push

