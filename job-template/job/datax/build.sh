#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/datax:20240501-amd64 -f job/datax/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/datax:20240501-amd64




