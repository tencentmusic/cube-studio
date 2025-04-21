#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/model_download:20240501 -f job/model_download/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/model_download:20240501



