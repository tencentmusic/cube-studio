pytorch在k8s之上的分布式训练

使用该模板能够帮助你在k8s自动创建一个pytorch分布式训练的集群。但是前提是需要你按照pytorch官方的方案先将代码编写为分布式形式。

# 单机版示例

https://github.com/pytorch/examples/blob/master/mnist/main.py

# 分布式版示例

https://github.com/kubeflow/pytorch-operator/blob/master/examples/mnist/mnist.py