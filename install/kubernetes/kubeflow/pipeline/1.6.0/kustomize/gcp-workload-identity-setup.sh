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

set -e

# Kubernetes Namespace
NAMESPACE=${NAMESPACE:-kubeflow}

# Google service Account (GSA)
SYSTEM_GSA=${SYSTEM_GSA:-$RESOURCE_PREFIX-kfp-system}
USER_GSA=${USER_GSA:-$RESOURCE_PREFIX-kfp-user}

# Kubernetes Service Account (KSA)
# Note, if deploying manifests/kustomize/env/gcp, you can add the following KSAs
# to the array of SYSTEM_KSA:
# * kubeflow-pipelines-minio-gcs-gateway needs gcs permissions
# * kubeflow-pipelines-cloudsql-proxy needs cloudsql permissions
SYSTEM_KSA=(ml-pipeline-ui ml-pipeline-visualizationserver)
USER_KSA=(pipeline-runner kubeflow-pipelines-container-builder kubeflow-pipelines-viewer)

if [ -n $USE_GCP_MANAGED_STORAGE ]; then
  SYSTEM_KSA+=(kubeflow-pipelines-minio-gcs-gateway)
  SYSTEM_KSA+=(kubeflow-pipelines-cloudsql-proxy)
fi

cat <<EOF

This script sets up Google service accounts, Kubernetes service accounts and workload identity bindings for a Kubeflow Pipelines (KFP) standalone deployment.
You can also choose to manually set these up based on documentation: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity.

Before you begin, please check the following list:
* Please first review introduction to workload identity: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity.
* KFP is already or will be deployed by standalone deployment: https://www.kubeflow.org/docs/pipelines/installation/standalone-deployment/
* gcloud is configured following steps: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity#before_you_begin.
* kubectl talks to the cluster KFP is deployed to: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl.
* The namespace you specified by NAMESPACE env var already exists on the cluster. You can create it by "kubectl create namespace \$NAMESPACE".

The following resources will be created or updated to create workload identity bindings between GSAs and KSAs:
* Google service accounts (GSAs)
* Service account IAM policy bindings on these GSAs
* Kubernetes service accounts with annotations in namespace "$NAMESPACE".

Note, this script is designed to be idempotent. If something went wrong, you can safely fix the error and rerun this script.

EOF

function usage {
cat <<\EOF
Usage:
```
PROJECT_ID=<your-gcp-project-id> RESOURCE_PREFIX=<your-chosen-prefix> NAMESPACE=<your-k8s-namespace> ./gcp-workload-identity-setup.sh
```

PROJECT_ID: GCP project ID your cluster belongs to.
RESOURCE_PREFIX: Your preferred resource prefix for GCP resources this script creates.
NAMESPACE: Optional. Kubernetes namespace your Kubeflow Pipelines standalone deployment belongs to. (Defaults to kubeflow)
USE_GCP_MANAGED_STORAGE: Optional. Defaults to "false", specify "true" if you intend to use GCP managed storage (Google Cloud Storage and Cloud SQL) following instructions in:
https://github.com/kubeflow/pipelines/tree/master/manifests/kustomize/sample
EOF
}
if [ -z "$PROJECT_ID" ]; then
  usage
  echo
  echo "Error: PROJECT_ID env variable is empty!"
  exit 1
fi
if [ -z "$RESOURCE_PREFIX" ]; then
  usage
  echo
  echo "Error: RESOURCE_PREFIX env variable is empty!"
  exit 1
fi
echo "Env variables set:"
echo "* PROJECT_ID=$PROJECT_ID"
echo "* RESOURCE_PREFIX=$RESOURCE_PREFIX"
echo "* NAMESPACE=$NAMESPACE"
echo "* USE_GCP_MANAGED_STORAGE=${USE_GCP_MANAGED_STORAGE:-false}"
echo

SYSTEM_GSA_FULL="$SYSTEM_GSA@$PROJECT_ID.iam.gserviceaccount.com"
USER_GSA_FULL="$USER_GSA@$PROJECT_ID.iam.gserviceaccount.com"

cat <<EOF

The following resources will be created or updated to create workload identity bindings between GSAs and KSAs:
* Google service accounts (GSAs):
  * $SYSTEM_GSA_FULL
  * $USER_GSA_FULL
* Service account IAM policy bindings on these GSAs to grant "Workload Identity User" role.
* Kubernetes service accounts with annotations in namespace "$NAMESPACE".
* $SYSTEM_GSA_FULL will be bound to these KSAs:
  ${SYSTEM_KSA[@]}.
* $USER_GSA_FULL will be bound to these KSAs:
  ${USER_KSA[@]}.

Note: if you prefer more granular workload identity bindings, you can modify this script to suit your needs.

EOF

read -p "Continue? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  exit 0
fi

echo "Creating Google service accounts..."
function create_gsa_if_not_present {
  local name=${1}
  local already_present=$(gcloud iam service-accounts list --filter='name:'$name'' --format='value(name)')
  if [ -n "$already_present" ]; then
    echo "Service account $name already exists"
  else
    gcloud iam service-accounts create $name
  fi
}
create_gsa_if_not_present $SYSTEM_GSA
create_gsa_if_not_present $USER_GSA

function create_ksa_if_not_present {
  local name=${1}
  if kubectl get serviceaccount $name -n $NAMESPACE >/dev/null; then
    echo "KSA $name already exists"
  else
    kubectl create serviceaccount $name -n $NAMESPACE --save-config
    echo "KSA $name created"
  fi
}

# Bind KSA to GSA through workload identity.
# Documentation: https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity
function bind_gsa_and_ksa {
  local gsa=${1}
  local ksa=${2}

  gcloud iam service-accounts add-iam-policy-binding $gsa@$PROJECT_ID.iam.gserviceaccount.com \
    --member="serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/$ksa]" \
    --role="roles/iam.workloadIdentityUser" \
    > /dev/null # hide verbose output

  create_ksa_if_not_present $ksa
  kubectl annotate serviceaccount \
    --namespace $NAMESPACE \
    --overwrite \
    $ksa \
    iam.gke.io/gcp-service-account=$gsa@$PROJECT_ID.iam.gserviceaccount.com
  echo "* Bound KSA $ksa to GSA $gsa"
}

echo "Binding each kfp system KSA to $SYSTEM_GSA"
for ksa in ${SYSTEM_KSA[@]}; do
  bind_gsa_and_ksa $SYSTEM_GSA $ksa
done

echo "Binding each kfp user KSA to $USER_GSA"
for ksa in ${USER_KSA[@]}; do
  bind_gsa_and_ksa $USER_GSA $ksa
done

echo
echo "All the workload identity bindings have succeeded!"
cat <<EOF


=============
Next steps:
* This script won't add IAM policies to grant these GSAs with permissions KFP needs, you need to do that by yourself.

### If **NOT** using GCP managed storage, you can:
* Give $SYSTEM_GSA_FULL "Storage Object Viewer" role to allow KFP UI load data in GCS in the same project:
gcloud projects add-iam-policy-binding $PROJECT_ID \\
  --member="serviceAccount:$SYSTEM_GSA_FULL" \\
  --role="roles/storage.objectViewer"

* Give $USER_GSA_FULL any permissions your pipelines, container builder and tensorboard need. For **QUICK** tryouts, you can give it Project Editor role for all permissions, but **WARNING** be aware this overgrants too much permission:
gcloud projects add-iam-policy-binding $PROJECT_ID \\
  --member="serviceAccount:$USER_GSA_FULL" \\
  --role="roles/editor"

### If using GCP managed storage, you **ALSO** need to give $SYSTEM_GSA_FULL these roles:
* "Storage Admin" role on specified GCS bucket to allow writing to specified GCS artifact bucket:
gsutil iam ch serviceAccount:$SYSTEM_GSA_FULL:roles/storage.admin gs://[BUCKET_NAME]

Or you can find other ways in https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add.

* "Cloud SQL Client" role to allow connecting to Cloud SQL instances:
gcloud projects add-iam-policy-binding $PROJECT_ID \\
  --member="serviceAccount:$SYSTEM_GSA_FULL" \\
  --role="roles/cloudsql.client"
EOF
