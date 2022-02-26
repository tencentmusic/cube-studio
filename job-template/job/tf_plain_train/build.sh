#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf2.3_plain_train:latest -f job/tf_plain_train/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf2.3_plain_train:latest