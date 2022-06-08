#!/bin/bash

set -ex

docker build -t ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601 -f job/video-audio/Dockerfile-cpu .
docker push ccr.ccs.tencentyun.com/cube-studio/video-audio:20210601





