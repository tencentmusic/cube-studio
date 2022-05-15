#!/bin/sh
mkdir -p /var/run/sshd && /usr/sbin/sshd && export PATH=$PATH:$1 && echo 'iam master and iam ready'
res=$(echo $?) && [ $res -ne 0 ] && echo "export error" && sleep 30 && exit -1

echo 'waiting for all worker ready...'
>> $1/machines
loop_time=0
while [ $loop_time -le 60 ];
do
  sleep 3 && echo "waiting for all worker ready..., loop_time=$loop_time"
  [ $(cat $1/machines | grep "^[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}$" | wc -l | awk '{print $1}') -ge $2 ] && break
  loop_time=$((loop_time+1))
done

cat $1/machines | grep "^[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}$" > $1/machines_tmp
cat $1/machines_tmp > $1/machines

res=$(echo $?) && [ $res -ne 0 ] && echo "timeout error" && sleep 30 && exit -1
[ $(wc $1/machines | awk '{print $1}') -lt $2 ] && echo "timeout error" && sleep 30 && exit -1

echo "start the job"
eval $3
res=$(echo $?) && [ $res -ne 0 ] && echo "job error" && sleep 30 && exit -1

sleep 30
