#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/pytorch_distributed_train_k8s:20201010 -f job/pytorch_distributed_train_k8s/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/pytorch_distributed_train_k8s:20201010


