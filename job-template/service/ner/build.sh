#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/ner:service-20220812 .
docker push ccr.ccs.tencentyun.com/cube-studio/ner:service-20220812



