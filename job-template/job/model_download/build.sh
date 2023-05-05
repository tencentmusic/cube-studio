#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/model_download:20221001 -f job/model_download/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/model_download:20221001



