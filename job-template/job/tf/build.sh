#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf:20230801 -f job/tf/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf:20230801


