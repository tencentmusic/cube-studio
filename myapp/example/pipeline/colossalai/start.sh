#!/bin/bash

echo "VK_TASK_INDEX" $VK_TASK_INDEX
echo "VC_TASK_INDEX" $VC_TASK_INDEX
echo "VC_WORKER_HOSTS" $VC_WORKER_HOSTS
echo "VC_WORKER_NUM" $VC_WORKER_NUM

# 安装环境
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 必须用他这个分支的代码
#git clone -b example/llama https://github.com/hpcaitech/ColossalAI.git
# 替换掉镜像中携带的
echo "PYTHONPATH=/mnt/admin/pipeline/example/colossalai/ColossalAI:$PYTHONPATH" >>/etc/profile
source /etc/profile
export PYTHONPATH=/mnt/admin/pipeline/example/colossalai/ColossalAI:$PYTHONPATH

# 基础环境安装比较久，镜像中已经携带了
# cd ColossalAI && CUDA_EXT=1 pip install . && pip install -r requirements.txt && pip install xformers
#
# 安装示例所需环境
pip install pytest datasets numpy transformers decorator tensorboard sentencepiece numpy tqdm psutil packaging pre-commit rich click fabric contexttimer ninja safetensors einops
# pip install torch -U

#启动ssh-server
apt install -y openssh-server
mkdir -p /var/run/sshd && /usr/sbin/sshd -D &
mkdir -p model checkpoint
# 先熟悉启动方式 https://colossalai.org/zh-Hans/docs/basics/launch_colossalai/
#判断是否是worker0
if [ $VC_TASK_INDEX -eq 0 ]; then
    echo "* This is rank 0"
    # 等待worker 的sshd启动
    sleep 10
    echo "* start multi-node training!"
    master_addr=`echo $VC_WORKER_HOSTS| cut -d ',' -f 1`
    echo $VC_WORKER_HOSTS | tr "," "\n"  > ~/.myhostfile

    # 启动分布式任务
#    colossalai run --nproc_per_node 1 --host $VC_WORKER_HOSTS --master_addr $master_addr pretrain.py \
#    --config 7b --dataset togethercomputer/RedPajama-Data-1T-Sample \
#    --num_epochs 1 --batch_size 2 --lr 3e-4 --weight_decay 0.1 \
#    --warmup_steps 2000 --max_length 2048 --mixed_precision fp16 \
#    --save_interval 1000 --save_dir model --tensorboard_dir tb_logs

    # 目前需要访问hugging face 下载模型和数据集，数据集1T。需要32个A800
    # cd /mnt/admin/pipeline/example/colossalai/ColossalAI/examples/language/llama && colossalai run --nproc_per_node 1 --hostfile ~/.myhostfile --master_addr $master_addr pretrain.py

    # 参考 https://github.com/hpcaitech/ColossalAI/tree/example/llama/examples/language/llama

else
    echo "* This is not rank 0"
    sleep 365d
fi
