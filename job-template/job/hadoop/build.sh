#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/hadoop:20221010 -f job/hadoop/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/hadoop:20221010


