# Sample installation

1. Prepare a cluster and setup kubectl context
Do whatever you want to customize your cluster. You can use existing cluster
or create a new one.
- **ML Usage** GPU normally is required for deep learning task.
You may consider create **zero-sized GPU node-pool with autoscaling**.
Please reference [GPU Tutorial](/samples/tutorials/gpu/).
- **Security** You may consider use **Workload Identity** in GCP cluster.

Here for simplicity, we create a small cluster with **--scopes=cloud-platform**
which grants all the GCP permissions to the cluster.

```
gcloud container clusters create mycluster \
  --zone us-central1-a \
  --machine-type n1-standard-2 \
  --scopes cloud-platform \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5 \
  --num-nodes 3
```

2. Prepare CloudSQL

Create CloudSQL instance. [Console](https://console.cloud.google.com/sql/instances).

Here is a sample for demo.

```
gcloud beta sql instances create mycloudsqlname \
  --database-version=MYSQL_5_7 \
  --tier=db-n1-standard-1 \
  --region=us-central1 \
  --root-password=password123
```

You may use **Private IP** to well protect your CloudSQL.
If you use **Private IP**, please go to [VPC network peering](https://console.cloud.google.com/networking/peering/list)
to double check whether the "cloudsql-mysql-googleais-com" is created and the "Exchange custom routes" is enabled. You
are expected to see "Peer VPC network is connected".

3. Prepare GCS Bucket

Create Cloud Storage bucket. [Console](https://console.cloud.google.com/storage).

```
gsutil mb -p myProjectId gs://myBucketName/
```

4. Customize your values
- Edit **params.env**, **params-db-secret.env** and **cluster-scoped-resources/params.env**
- Edit kustomization.yaml to set your namespace, e.x. "kubeflow"

5. (Optional.) If the cluster is on Workload Identity, please run **[gcp-workload-identity-setup.sh](../gcp-workload-identity-setup.sh)**
  The script prints usage documentation when calling without argument. Note, you should
  call it with `USE_GCP_MANAGED_STORAGE=true` env var.

  - make sure the Google Service Account (GSA) can access the CloudSQL instance and GCS bucket
  - if your workload calls other GCP APIs, make sure the GSA can access them

6. Install

```
kubectl apply -k sample/cluster-scoped-resources/

kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s

kubectl apply -k sample/
# If upper one action got failed, e.x. you used wrong value, try delete, fix and apply again
# kubectl delete -k sample/

kubectl wait applications/mypipeline -n kubeflow --for condition=Ready --timeout=1800s
```

Now you can find the installation in [Console](http://console.cloud.google.com/ai-platform/pipelines)
