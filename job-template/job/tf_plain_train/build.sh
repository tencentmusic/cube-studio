#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf2.3_plain_train:latest -f job/tf_plain_train/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf2.3_plain_train:latest