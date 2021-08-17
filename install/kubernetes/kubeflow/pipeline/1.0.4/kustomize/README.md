# Install Kubeflow Pipelines

This folder contains Kubeflow Pipelines Kustomize manifests for a light weight
deployment. You can follow the instruction and deploy Kubeflow Pipelines in an
existing cluster.

To install Kubeflow Pipelines, you have several options.
- Via [GCP AI Platform UI](http://console.cloud.google.com/ai-platform/pipelines).
- Via an upcoming commandline tool.
- Via Kubectl with Kustomize, it's detailed here.

## Install via Kustomize

Deploy latest version of Kubeflow Pipelines.

It uses following default settings.
- image: latest released images
- namespace: kubeflow
- application name: pipeline

### Option-1 Install it to any K8s cluster
It's based on in-cluster PersistentVolumeClaim storage.

```
kubectl apply -k cluster-scoped-resources/
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s
kubectl apply -k env/platform-agnostic/
kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Now you can access it via localhost:8080

### Option-2 Install it to GCP with in-cluster PersistentVolumeClaim storage
It's based on in-cluster PersistentVolumeClaim storage.
Additionally, it introduced a proxy in GCP to allow user easily access KFP safely.

```
kubectl apply -k cluster-scoped-resources/
kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s

kubectl apply -k env/dev/
kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s

# Or visit http://console.cloud.google.com/ai-platform/pipelines
kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com
```

### Option-3 Install it to GCP with CloudSQL & GCS-Minio managed storage
Its storage is based on CloudSQL & GCS. It's better than others for production usage.

Please following [sample](sample/README.md) for a customized installation.

### Option-4 Install it to AWS with S3 and RDS MySQL
Its storage is based on S3 & AWS RDS. It's more natural for AWS users to use this option.

Please following [AWS Instructions](env/aws/README.md) for installation.

Note: Community maintains a repo [e2fyi/kubeflow-aws](https://github.com/e2fyi/kubeflow-aws/tree/master/pipelines) for AWS.

## Uninstall

If the installation is based on CloudSQL/GCS, after the uninstall, the data is still there,
reinstall a newer version can reuse the data.

```
### 1. namespace scoped
# Depends on how you installed it:
kubectl kustomize env/platform-agnostic | kubectl delete -f -
# or
kubectl kustomize env/dev | kubectl delete -f -
# or
kubectl kustomize env/gcp | kubectl delete -f -
# or
kubectl delete applications/pipeline -n kubeflow

### 2. cluster scoped
kubectl delete -k cluster-scoped-resources/
```

## Troubleshooting

### Permission error installing Kubeflow Pipelines to a cluster

Run

```
kubectl create clusterrolebinding your-binding --clusterrole=cluster-admin --user=[your-user-name]
```

### Samples requires "user-gcp-sa" secret

If sample code requires a "user-gcp-sa" secret, you could create one by

-   First download the GCE VM service account token
    [Document](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys)

```
gcloud iam service-accounts keys create application_default_credentials.json \
  --iam-account [SA-NAME]@[PROJECT-ID].iam.gserviceaccount.com
```

-   Run

```
kubectl create secret -n [your-namespace] generic user-gcp-sa --from-file=user-gcp-sa.json=application_default_credentials.json`
```
