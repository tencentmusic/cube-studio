#!/bin/bash
#
# Copyright 2019 The Kubeflow Authors
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

function create_gsa_if_not_present {
  local name=${1}
  local already_present=$(gcloud iam service-accounts list --filter='name:'$name'' --format='value(name)')
  if [ -n "$already_present" ]; then
    echo "Service account $name already exists"
  else
    gcloud iam service-accounts create $name
  fi
}

# Bind KSA to GSA through workload identity.
# Documentation: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity
function bind_gsa_and_ksa {
  local gsa=${1}
  local ksa=${2}
  local project=${3:-$PROJECT_ID}
  local gsa_full="$gsa@$project.iam.gserviceaccount.com"
  local namespace=${4:-$NAMESPACE}

  gcloud iam service-accounts add-iam-policy-binding $gsa_full \
    --member="serviceAccount:$project.svc.id.goog[$namespace/$ksa]" \
    --role="roles/iam.workloadIdentityUser" \
    > /dev/null # hide verbose output
  kubectl annotate serviceaccount \
    --namespace $namespace \
    --overwrite \
    $ksa \
    iam.gke.io/gcp-service-account=$gsa_full
  echo "* Bound KSA $ksa in namespace $namespace to GSA $gsa_full"
}

# This can be used to programmatically verify workload identity binding grants corresponding GSA
# permissions successfully.
# Usage: verify_workload_identity_binding $KSA $NAMESPACE
#
# If you want to verify manually, use the following command instead:
# kubectl run test-$RANDOM --rm -it --restart=Never \
#     --image=google/cloud-sdk:slim \
#     --serviceaccount $ksa \
#     --namespace $namespace \
#     -- /bin/bash
# It connects you to a pod using specified KSA running an image with gcloud and gsutil CLI tools.
function verify_workload_identity_binding {
  local ksa=${1}
  local namespace=${2}
  local max_attempts=10
  local workload_identity_is_ready=false
  for i in $(seq 1 ${max_attempts})
  do
    workload_identity_is_ready=true
    kubectl run test-$RANDOM --rm -i --restart=Never \
        --image=google/cloud-sdk:slim \
        --serviceaccount $ksa \
        --namespace $namespace \
        -- gcloud auth list || workload_identity_is_ready=false
    kubectl run test-$RANDOM --rm -i --restart=Never \
        --image=google/cloud-sdk:slim \
        --serviceaccount $ksa \
        --namespace $namespace \
        -- gsutil ls gs:// || workload_identity_is_ready=false
    if [ "$workload_identity_is_ready" = true ]; then
      break
    fi
  done
  if [ ! "$workload_identity_is_ready" = true ]; then
    echo "Workload identity bindings are not ready after $max_attempts attempts"
    return 1
  fi
}
