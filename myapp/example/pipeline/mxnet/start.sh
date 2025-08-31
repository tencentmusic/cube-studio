#!/bin/bash

#pip install numpy
echo "MX_CONFIG" $MX_CONFIG
echo "DMLC_PS_ROOT_PORT" $DMLC_PS_ROOT_PORT
echo "DMLC_PS_ROOT_URI" $DMLC_PS_ROOT_URI
echo "DMLC_NUM_SERVER" $DMLC_NUM_SERVER
echo "DMLC_NUM_WORKER" $DMLC_NUM_WORKER
echo "DMLC_ROLE" $DMLC_ROLE
echo "DMLC_USE_KUBERNETES" $DMLC_USE_KUBERNETES


DMLC_ROLE=$(echo $DMLC_ROLE)
cp train_mnist.py /incubator-mxnet/example/image-classification/train_mnist.py
if [ "$DMLC_ROLE" == "scheduler" ] || [ "$DMLC_ROLE" == "server" ]; then
  echo "finish"
  # python /incubator-mxnet/example/image-classification/train_mnist.py
elif [ "$DMLC_ROLE" == "worker" ]; then
  echo "finish"
  # python /incubator-mxnet/example/image-classification/train_mnist.py --num-epochs 10 --num-layers 2 --kv-store dist_device_sync --gpus 0
else
  echo "DMLC_ROLE is not set or has an invalid value."
fi

