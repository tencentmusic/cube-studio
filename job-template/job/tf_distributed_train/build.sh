#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf_distributed_train:latest -f job/tf_distributed_train/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf_distributed_train:latest


