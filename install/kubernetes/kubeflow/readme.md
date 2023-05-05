
# 基础环境需求
k8s集群，分布式存储，这些都假设在前面的步骤已经完成

# 部署train-operator

支持 [TensorFlow/PyTorch/Apache MXNet/XGBoost/MPI jobs](https://github.com/kubeflow/training-operator/tree/v1.4.0)

```bash
kubectl apply -k train-operator/manifests/overlays/standalone
```

