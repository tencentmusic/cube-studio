#!/bin/bash

#set -ex

mkdir -p /var/run/sshd && /usr/sbin/sshd && export PATH=$PATH:$1
echo 0 > $1/${MY_POD_IP}.lock
>> $1/machines
while [ -z $(grep "${MY_POD_IP}" $1/machines) ];do echo "$MY_POD_IP" >> $1/machines; sleep 1; done;
echo "iam worker $MY_POD_IP and iam ready"

res=$(echo $?) && [ $res -ne 0 ] && echo "init error" && sleep 30 && exit -1
sleep 36000000
