#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/volcano:20211001 -f job/volcano/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/volcano:20211001



