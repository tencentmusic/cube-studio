#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1 -f job/sklearn_estimator/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/sklearn_estimator:v1

