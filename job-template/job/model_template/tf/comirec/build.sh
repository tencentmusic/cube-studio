#!/bin/bash
set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/tf2.3_model_template_comirec:latest -f job/model_template/tf/comirec/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/tf2.3_model_template_comirec:latest