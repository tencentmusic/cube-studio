#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf_distributed_eval:latest -f job/tf_distributed_evaluation/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf_distributed_eval:latest


