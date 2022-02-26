#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf_model_offline_predict:latest -f job/tf_model_offline_predict/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf_model_offline_predict:latest