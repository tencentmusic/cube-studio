#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf_distributed_train_k8s:20221010 -f job/tf_distributed_train_k8s/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf_distributed_train_k8s:20221010


