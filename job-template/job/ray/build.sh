#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/ray:cpu-20210601 -f job/ray/Dockerfile-cpu .
docker push ai.tencentmusic.com/tme-public/ray:cpu-20210601

docker build -t ai.tencentmusic.com/tme-public/ray:gpu-20210601 -f job/ray/Dockerfile-gpu .
docker push ai.tencentmusic.com/tme-public/ray:gpu-20210601



