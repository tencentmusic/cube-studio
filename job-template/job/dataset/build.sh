#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/datax:20220601 -f job/datax/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/datax:20220601



