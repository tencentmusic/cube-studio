#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/volcano:offline-predict-20220101 -f job/model_offline_predict/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/volcano:offline-predict-20220101




