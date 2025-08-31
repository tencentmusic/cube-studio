#!/bin/bash

#pip install numpy
echo "PADDLE_JOB_ID" $PADDLE_JOB_ID
echo "PADDLE_NNODES" $PADDLE_NNODES
echo "PADDLE_MASTER" $PADDLE_MASTER
echo "POD_IP_DUMMY" $POD_IP_DUMMY

# python -m paddle.distributed.launch run_check
