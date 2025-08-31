#!/bin/bash

echo "VK_TASK_INDEX" $VK_TASK_INDEX
echo "VC_TASK_INDEX" $VC_TASK_INDEX
echo "VC_WORKER_HOSTS" $VC_WORKER_HOSTS
echo "VC_WORKER_NUM" $VC_WORKER_NUM

# 安装环境
#pip install numpy

#启动ssh-server
sudo mkdir -p /var/run/sshd && sudo /usr/sbin/sshd -D &

# 判断是否是worker0,worker0做分配任务，所有worker做任务执行
if [ $VC_TASK_INDEX -eq 0 ]; then
    sudo echo "* This is rank 0"
    # 等待worker 的sshd启动
    sleep 10
    #编写deepspeed需要的hostfile文件
    echo $VC_WORKER_HOSTS | tr "," "\n" | sed 's/$/ slots=1/' >> ~/.myhostfile
    echo "* start multi-node training!"
    FILE=cifar-10-python.tar.gz

    if [ ! -f "$FILE" ]; then
      wget -O $FILE https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    fi
    # pipeline运行
    #deepspeed -H ~/.myhostfile main.py --data_path /mnt/xxx/rm-static --model_name_or_path /mnt/xx/model_zoo/RLHF/llama-7b-hf --num_padding_at_beginning 0 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_seq_len 512 --learning_rate 2e-5 --weight_decay 0.1 --num_train_epochs 1 --offload --gradient_accumulation_steps 4 --gradient_checkpointing --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --zero_stage 2 --data_split 0,8,2 --deepspeed --output_dir /mnt/xx/model_zoo/result/llama &> /mnt/xx/model_zoo/result/training_llama.log
     deepspeed -H ~/.myhostfile train.py --deepspeed_config=ds_config.json -p $VC_WORKER_NUM --steps=20

else
    echo "* This is not rank 0，sleep until job finish"
    sleep 365d
fi


