#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/object_detection_on_darknet:v1 -f Dockerfile  .
docker push ccr.ccs.tencentyun.com/cube-studio/object_detection_on_darknet:v1

