#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf2.3_model_evaluation:latest -f job/tf_model_evaluation/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf2.3_model_evaluation:latest