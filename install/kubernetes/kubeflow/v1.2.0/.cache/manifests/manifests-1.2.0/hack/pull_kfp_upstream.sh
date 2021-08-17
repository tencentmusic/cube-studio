#!/usr/bin/env bash

# Usage:
# This script should be run from kubeflow/manifest repo's root:
# ./hack/pull_kfp_upstream.sh

set -ex

# Please edit the following version before running the script to pull new
# pipelines version.
export PIPELINES_VERSION=1.0.4
export PIPELINES_SRC_REPO=https://github.com/kubeflow/pipelines.git

if [ -d pipeline/upstream ]; then
    # Updates
    kpt pkg update pipeline/upstream/@$PIPELINES_VERSION --strategy force-delete-replace
else
    # Pulling for the first time
    kpt pkg get $PIPELINES_SRC_REPO/manifests/kustomize@$PIPELINES_VERSION pipeline/upstream
fi

# Before kubeflow/pipelines/manifests/kustomize supports kustomize v3.5+, we
# have to convert kustomization.yaml env to envs syntax, so that it is compatible
# with latest kustomize used in kubeflow/manifests.
# ref: https://github.com/kubeflow/manifests/pull/1248#issuecomment-645739641
find pipeline/upstream -name 'kustomization.yaml' -exec \
    sed -i.bak 's#env: \(.*\)#envs: ["\1"]#g' {} +
