#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/model_register:20230501 -f job/model_register/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/model_register:20230501

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/model_register:20230501 -f job/model_register/Dockerfile . --push


