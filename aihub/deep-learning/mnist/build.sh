#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/mnist:20220814 -f Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/mnist:20220814



