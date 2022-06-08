#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf2.3_keras_train:latest -f job/tf_keras_train/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf2.3_keras_train:latest