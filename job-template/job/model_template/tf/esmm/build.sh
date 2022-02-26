#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf2.3_model_template_esmm:latest -f job/model_template/tf/esmm/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf2.3_model_template_esmm:latest


