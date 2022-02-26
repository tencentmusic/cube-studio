#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/volcano:20211001 -f job/volcano/Dockerfile .
docker push ai.tencentmusic.com/tme-public/volcano:20211001



