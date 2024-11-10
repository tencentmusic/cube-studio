#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/volcano:20230601 -f job/volcano/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/volcano:20230601



