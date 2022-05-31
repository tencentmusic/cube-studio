#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/deploy-service:20211001 -f job/deploy-service/Dockerfile .
docker push ai.tencentmusic.com/tme-public/deploy-service:20211001



