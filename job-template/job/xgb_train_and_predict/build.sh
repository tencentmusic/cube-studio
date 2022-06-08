#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/xgb_train_and_predict:v1 -f job/xgb_train_and_predict/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/xgb_train_and_predict:v1

