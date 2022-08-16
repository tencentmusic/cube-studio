#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/mxnet:20221010 -f job/mxnet/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/mxnet:20221010


