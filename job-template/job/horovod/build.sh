#!/bin/bash

set -ex
docker build -t ccr.ccs.tencentyun.com/cube-studio/horovod:20230801 -f job/horovod/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/horovod:20230801




