#!/bin/bash

set -ex

#docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/ray:cpu-20230801 -f job/ray/Dockerfile-cpu .
#docker push ccr.ccs.tencentyun.com/cube-studio/ray:cpu-20230801

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20250301 -f job/ray/Dockerfile-gpu .
docker push ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20250301



