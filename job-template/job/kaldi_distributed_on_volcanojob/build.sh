#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/kaldi_distributed_on_volcano:v2 -f job/kaldi_distributed_on_volcanojob/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/kaldi_distributed_on_volcano:v2



