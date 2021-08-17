# Sample installation

1. Create an EKS cluster and setup kubectl context

Using configuration file to simplify EKS cluster creation process:
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: kfworkshop
  region: us-west-2
  version: '1.17'
# If your region has multiple availability zones, you can specify 3 of them.
availabilityZones: ["us-west-2b", "us-west-2c", "us-west-2d"]

# NodeGroup holds all configuration attributes that are specific to a nodegroup
# You can have several node group in your cluster.
nodeGroups:
  - name: cpu-nodegroup
    instanceType: m5.xlarge
    desiredCapacity: 2
    minSize: 0
    maxSize: 4
    volumeSize: 50
    # ssh:
    #   allow: true
    #   publicKeyPath: '~/.ssh/id_rsa.pub'

  # Example of GPU node group
  - name: Tesla-V100
    instanceType: p3.8xlarge
    # Make sure the availability zone here is one of cluster availability zones.
    availabilityZones: ["us-west-2b"]
    desiredCapacity: 0
    minSize: 0
    maxSize: 4
    volumeSize: 50
    # ssh:
    #   allow: true
    #   publicKeyPath: '~/.ssh/id_rsa.pub'
```
Run this command to create EKS cluster
```
eksctl create cluster -f cluster.yaml
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

kubectl wait crd/applications.app.k8s.io --for condition=established --timeout=60s

kubectl apply -k ./
# If upper one action got failed, e.x. you used wrong value, try delete, fix and apply again
# kubectl delete -k ./

kubectl wait applications/mypipeline -n kubeflow --for condition=Ready --timeout=1800s

kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Now you can access via `localhost:8080`
