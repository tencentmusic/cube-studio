# Sample installation

1. Create an EKS cluster

Run this command to create EKS cluster
```
eksctl create cluster \
--name AWS-KFP \
--version 1.17 \
--region us-west-2 \
--nodegroup-name linux-nodes \
--node-type m5.xlarge \
--nodes 2 \
--nodes-min 1 \
--nodes-max 4 \
--managed
```

2. Prepare S3

Create S3 bucket. [Console](https://console.aws.amazon.com/s3/home).

Run this command to create S3 bucket by changing `<YOUR_S3_BUCKET_NAME>` to your prefer s3 bucket name.

```
export S3_BUCKET=<YOUR_S3_BUCKET_NAME>
export AWS_REGION=us-west-2
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
```

3. Prepare RDS

Follow this [doc](https://www.kubeflow.org/docs/aws/rds/#deploy-amazon-rds-mysql-in-your-environment) to set up AWS RDS instance.

4. Customize your values
- Edit [params.env](params.env), [secret.env](secret.env) and [minio-artifact-secret-patch.env](minio-artifact-secret-patch.env)

5. Install

```
kubectl apply -k ../../cluster-scoped-resources
# If upper one action got failed, e.x. you used wrong value, try delete, fix and apply again
# kubectl delete -k ../../cluster-scoped-resources

kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s

kubectl apply -k ./
# If upper one action got failed, e.x. you used wrong value, try delete, fix and apply again
# kubectl delete -k ./

kubectl wait applications/pipeline -n kubeflow --for condition=Ready --timeout=1800s

kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Now you can access via `localhost:8080`
