#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf_distributed_eval:latest -f job/tf_distributed_evaluation/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf_distributed_eval:latest


