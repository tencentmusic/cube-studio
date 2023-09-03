#!/bin/bash

set -ex

#docker build -t ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001 -f job/deploy-service/Dockerfile .
#docker push ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001

docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/deploy-service:20230501 -f job/deploy-service/Dockerfile . --push

