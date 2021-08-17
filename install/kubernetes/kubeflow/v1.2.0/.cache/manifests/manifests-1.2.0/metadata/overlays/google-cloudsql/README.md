This directory contains configurations and guidelines on setting up metadata services to connect to a [Google CloudSQL](https://cloud.google.com/sql) instance.
You will get all the benefits of using CloudSQL comparing to managing your own MySQL server in a Kubernetes cluster.

#### Prerequisites
- Install [kustomize](https://github.com/kubernetes-sigs/kustomize) for building Kubernetes configurations.
- Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for managing workloads on Kubernetes clusters.

#### 0. Remove default metadata services.
By default, Metadata component starts a MySQL server in `kubeflow` namespace. Since we are going to deploy metadata services with CloudSQL, you should delete the default services by running

```
kustomize build metadata/overlays/db | kubectl delete -n kubeflow -f -
```

#### 1. Create a CloudSQL instance.

If you don't have an existing one, you need to [create a CloudSQL instance](https://cloud.google.com/sql/docs/mysql/create-instance) of type MySQL in your GCP project.
If you want to connect the instance via private IP, you also need to enable the private IP configuration when creating the instance.

#### 2. Create a Kubernetes secret for accessing the CloudSQL instance.
You can follow [this guide](https://cloud.google.com/sql/docs/mysql/connect-kubernetes-engine#secrets)
to set up a [service account with permissions](https://cloud.google.com/sql/docs/mysql/sql-proxy#create-service-account) to connect to the instance, download the JSON key file, and name it `credentials.json`.
You need to create a secret via command:
```
kubectl create secret -n kubeflow generic cloudsql-instance-credentials --from-file <local_path>/credentials.json
```
Note that you must name the key file `credentials.json`, because we will later refer to this file name in the deployment configuration.

#### 3. Create a Kubernetes secret for MySQL account and password.
Besides the service account with permissions, the metadata services also need a MySQL account name and password to be authenticated for accessing databases. Secret is the way how Kubernetes manages sensitive information.

You need to [create a secret](https://kubernetes.io/docs/concepts/configuration/secret/#creating-your-own-secrets) under `kubeflow` namespace with name `metadata-db-secrets`, containing values of `MYSQL_USERNAME` and `MYSQL_PASSWORD`.
You should be able to see the secret after its creation via command:
```
kubectl describe secrets -n kubeflow metadata-db-secrets

Name:         metadata-db-secrets
Namespace:    kubeflow
Labels:       kustomize.component=metadata
Annotations:  
Type:         Opaque

Data
====
MYSQL_PASSWORD:  9 bytes
MYSQL_USERNAME:  4 bytes
```

#### 4. Specify the instance connection name.
Change the value of `MYSQL_INSTANCE` in `params.env` to your CloudSQL instance connection name. The connection name is in the form of `<project-id>:<region>:<instance-id>`.

#### 5. Start metadata services with CloudSQL proxy.
Start metadata services with CloudSQL proxy sidecar containers via command:
```
kustomize build metadata/overlays/google-cloudsql | kubectl apply -n kubeflow -f -
```
You may find the CloudSQL proxy container logs useful to debug connection errors.
 
