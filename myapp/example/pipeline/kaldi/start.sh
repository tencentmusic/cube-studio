#!/bin/bash
# 安装示例所需环境
#pip install numpy

#启动ssh-server
mkdir -p /var/run/sshd && /usr/sbin/sshd -D &
cp -r /opt/kaldi/kaldi ./
root_path=$PWD/kaldi/egs/aishell2/s5
mkdir -p $root_path/$KFJ_RUN_ID && cp k8s.pl $root_path/$KFJ_RUN_ID/ && echo 0 > $root_path/$KFJ_RUN_ID/${MY_POD_IP}.lock


#判断是否是worker0
if [ $VC_TASK_INDEX -eq 0 ]; then
    echo "* This is rank 0"
    # 等待worker 的sshd启动
    sleep 10
    echo "* start multi-node training!"
    master_addr=`echo $VC_WORKER_HOSTS| cut -d ',' -f 1`
    echo $VC_WORKER_HOSTS | tr "," "\n"  > $root_path/$KFJ_RUN_ID/machines
    sleep 10000000

    cd $root_path && bash run.sh

else
    echo "* This is not rank 0"
    sleep 365d
fi
