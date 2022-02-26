#!/bin/bash

set -ex

docker build -t ai.tencentmusic.com/tme-public/video-audio:20210601 -f job/video-audio/Dockerfile-cpu .
docker push ai.tencentmusic.com/tme-public/video-audio:20210601





