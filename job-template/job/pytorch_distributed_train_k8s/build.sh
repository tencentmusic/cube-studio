#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/pytorch_distributed_train_k8s:20201010 -f job/pytorch_distributed_train_k8s/Dockerfile .
docker push ai.tencentmusic.com/tme-public/pytorch_distributed_train_k8s:20201010


