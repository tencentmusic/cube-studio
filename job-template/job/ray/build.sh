#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/ray:cpu-20210601 -f job/ray/Dockerfile-cpu .
docker push ccr.ccs.tencentyun.com/cube-studio/ray:cpu-20210601

docker build -t ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20210601 -f job/ray/Dockerfile-gpu .
docker push ccr.ccs.tencentyun.com/cube-studio/ray:gpu-20210601



