#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf_distributed_train:latest -f job/tf_distributed_train/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf_distributed_train:latest


