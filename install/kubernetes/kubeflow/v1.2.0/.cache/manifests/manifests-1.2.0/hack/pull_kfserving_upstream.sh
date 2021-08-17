#!/usr/bin/env bash

# Usage:
# This script should be run from kubeflow/manifest repo's root:
# ./hack/pull_kfserving_upstream.sh

set -ex

# Please edit the following version before running the script to pull new
# kfserving version.
export KFSERVING_VERSION=v0.4.1
export KFSERVING_SRC_REPO=https://github.com/kubeflow/kfserving.git

if [ -d kfserving/upstream ]; then
    rm -rf kfserving/upstream
fi
kpt pkg get $KFSERVING_SRC_REPO/install/$KFSERVING_VERSION kfserving/upstream

# Replace 'kfserving-system' namespace references with 'kubeflow'.
find kfserving/upstream -name 'kfserving.yaml' -exec \
    sed -i.bak 's/kfserving-system/kubeflow/' {} +
