#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/dataset:20230401 -f job/dataset/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/dataset:20230401



