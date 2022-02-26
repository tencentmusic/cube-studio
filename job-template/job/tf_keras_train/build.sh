#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf2.3_keras_train:latest -f job/tf_keras_train/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf2.3_keras_train:latest