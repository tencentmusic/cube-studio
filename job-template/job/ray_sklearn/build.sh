#!/bin/bash

set -ex

docker build --network=host -t ai.tencentmusic.com/tme-public/sklearn_estimator:v1 -f job/sklearn_estimator/Dockerfile .
docker push ai.tencentmusic.com/tme-public/sklearn_estimator:v1

