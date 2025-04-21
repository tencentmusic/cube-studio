#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/ray-sklearn:20230801 -f job/ray_sklearn/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/ray-sklearn:20230801

