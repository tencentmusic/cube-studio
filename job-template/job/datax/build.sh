#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/datax:20220601 -f job/datax/Dockerfile .
docker push ai.tencentmusic.com/tme-public/datax:20220601



