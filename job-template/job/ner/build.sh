#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/ner:20220812 -f job/ner/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/ner:20220812



