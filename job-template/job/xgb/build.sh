#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/xgb:20230801 -f job/xgb/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/xgb:20230801

