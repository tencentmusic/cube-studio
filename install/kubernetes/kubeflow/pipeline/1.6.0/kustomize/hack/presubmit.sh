#!/bin/bash
#
# Copyright 2021 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a wrapper script on top of test.sh, it installs required dependencies.

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd)"
TMP="$(mktemp -d)"

pushd "${TMP}"
# Install Kustomize
KUSTOMIZE_VERSION=3.10.0
# Reference: https://kubectl.docs.kubernetes.io/installation/kustomize/binaries/
curl -s -O "https://raw.githubusercontent.com/\
kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"
chmod +x install_kustomize.sh
./install_kustomize.sh "${KUSTOMIZE_VERSION}" /usr/local/bin/

# Reference: https://github.com/mikefarah/yq/releases/tag/3.4.1
curl -s -LO "https://github.com/mikefarah/yq/releases/download/3.4.1/yq_linux_amd64"
chmod +x yq_linux_amd64
mv yq_linux_amd64 /usr/local/bin/yq
popd

# kpt and kubectl should already be installed in gcr.io/google.com/cloudsdktool/cloud-sdk:latest
# so we do not need to install them here

# trigger real unit tests
${DIR}/test.sh
# verify release script runs properly

${DIR}/release.sh v1.2.3-dummy
# --no-pager sends output to stdout
# Show git diff, so people can manually verify results of the release script
git --no-pager diff
