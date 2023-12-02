#!/bin/bash

set -ex
#
docker build -t ccr.ccs.tencentyun.com/cube-studio/datax:20230601 -f job/datax/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/datax:20230601
#
#docker buildx build --platform linux/amd64 -t ccr.ccs.tencentyun.com/cube-studio/datax:20230501 -f job/datax/Dockerfile . --push



