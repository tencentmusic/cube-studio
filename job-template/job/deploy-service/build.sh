#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001 -f job/deploy-service/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001



