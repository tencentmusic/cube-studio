#!/bin/bash

set -ex

docker build --network=host -t ai.tencentmusic.com/tme-public/object_detection_on_darknet:v1 -f Dockerfile  .
docker push ai.tencentmusic.com/tme-public/object_detection_on_darknet:v1

