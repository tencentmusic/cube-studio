#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/model:20221001 -f job/register_model/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/model:20221001



