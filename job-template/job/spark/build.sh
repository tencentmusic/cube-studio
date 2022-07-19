#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/spark:20221010 -f job/spark/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/spark:20221010


