#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/deploy-service:20240601 -f job/deploy-service/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/deploy-service:20240601

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/deploy-service:20240601 -f job/deploy-service/Dockerfile . --push

