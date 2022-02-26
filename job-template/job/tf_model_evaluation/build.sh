#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf2.3_model_evaluation:latest -f job/tf_model_evaluation/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf2.3_model_evaluation:latest