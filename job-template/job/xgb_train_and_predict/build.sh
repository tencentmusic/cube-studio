#!/bin/bash

set -ex

docker build --network=host -t ai.tencentmusic.com/tme-public/xgb_train_and_predict:v1 -f job/xgb_train_and_predict/Dockerfile .
docker push ai.tencentmusic.com/tme-public/xgb_train_and_predict:v1

