#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/ner-service:20220812 -f job/ner_service/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/ner-service:20220812



