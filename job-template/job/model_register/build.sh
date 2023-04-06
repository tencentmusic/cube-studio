#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/model_register:20221001 -f job/model_register/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/model_register:20221001



