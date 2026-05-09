#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/volcano:20250801 -f job/volcano/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/volcano:20250801



