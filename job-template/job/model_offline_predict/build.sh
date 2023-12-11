#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/offline-predict:20230801 -f job/model_offline_predict/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/offline-predict:20230801




