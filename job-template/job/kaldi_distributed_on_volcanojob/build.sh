#!/bin/bash

set -ex

docker build --network=host -t ai.tencentmusic.com/tme-public/kaldi_distributed_on_volcano:v2 -f job/kaldi_distributed_on_volcanojob/Dockerfile .
docker push ai.tencentmusic.com/tme-public/kaldi_distributed_on_volcano:v2



