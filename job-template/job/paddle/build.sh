#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/paddle:20230801 -f job/paddle/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/paddle:20230801


