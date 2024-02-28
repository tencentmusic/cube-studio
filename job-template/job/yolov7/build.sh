#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01 -f Dockerfile  .
docker push ccr.ccs.tencentyun.com/cube-studio/yolov7:2024.01

