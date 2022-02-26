#!/bin/bash
set -ex

docker build -t ai.tencentmusic.com/tme-public/tf2.3_model_template_comirec:latest -f job/model_template/tf/comirec/Dockerfile .
docker push ai.tencentmusic.com/tme-public/tf2.3_model_template_comirec:latest