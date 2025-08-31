#!/bin/bash

# ip列表
IP_LIST=(
106.15.42.11
139.224.197.39
139.224.67.155
106.15.75.152
139.224.54.32
47.100.68.189
139.196.255.114
139.224.55.87
139.224.215.131
47.100.72.14
)

USERNAME="root"
PASSWORD="1qaz2wsx#EDC"

for IP in "${IP_LIST[@]}"
do
    echo "正在连接 $IP..."
    # 第一步 先免密
#    sshpass -p "$PASSWORD" ssh-copy-id -o StrictHostKeyChecking=no $USERNAME@$IP

    # 后面持续
    scp init.sh  $USERNAME@$IP:/home/ &
    ssh $USERNAME@$IP "echo $IP && export STAGE=44 && bash /home/init.sh" &

    echo "在 $IP 上执行完成"
done

wait

# 最后为机器统一加标签
# kubectl label nodes --all train=true cpu=true notebook=true service=true org=public --overwrite


# 可以压测
#import os
#
#for pipeline_id in range(17,116):
#    command = f'curl -H "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjdWJlLXN0dWRpbyIsInN1YiI6ImFkbWluIn0.z4XJRqUI4v39MUYDUKdIuQsP5QlRENyVkQIp6a-1fb0" http://106.14.32.167/pipeline_modelview/api/run_pipeline/{pipeline_id}'
#    os.system(command)
#
#for notebook_id in range(145,253):
#    command = f'curl -H "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjdWJlLXN0dWRpbyIsInN1YiI6ImFkbWluIn0.z4XJRqUI4v39MUYDUKdIuQsP5QlRENyVkQIp6a-1fb0" http://106.14.32.167/notebook_modelview/api/reset/{notebook_id}'
#    os.system(command)

