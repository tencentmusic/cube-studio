#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf_model_offline_predict:latest -f job/tf_model_offline_predict/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf_model_offline_predict:latest