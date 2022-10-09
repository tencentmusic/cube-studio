#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/mnist:svc-20220817 -f Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/mnist:svc-20220817



