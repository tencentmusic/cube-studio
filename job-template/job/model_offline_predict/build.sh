#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/volcano:offline-predict-20220101 -f job/volcano_predict/Dockerfile .
docker push ai.tencentmusic.com/tme-public/volcano:offline-predict-20220101




