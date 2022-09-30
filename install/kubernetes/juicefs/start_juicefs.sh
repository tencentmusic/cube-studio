#记得提前修改.env中的ip地址，将其修改为自己的ip
source .env
#通过docker-compose启动用于元数据存储、块对象存储的redis和minio；
docker-compose up -d
#安装juicefs
cp -r juicefs /usr/local/bin && chmod 777 -R /usr/local/bin/juicefs
#格式化文件系统,juicesfs支持将不同的redis database以及minio bucket格式化成不同的文件系统
juicefs format \
    --storage minio \
    #--bucket http://10.48.60.91:9010/<bucket> \
    --bucket http://${JUICEFS_HOST_IP}:9010/juicefs \
    --access-key root \
    --secret-key Dewe_2131 \
    #"redis://:myredispassword@10.48.60.91:6382/<database>" \
    "redis://:${REDIS_PASSWORD}@${JUICEFS_HOST_IP}:6382/1" \
    myjfs

#安装juicefs的驱动
for i in $(ls juicefs-decive-of-k8s/); do kubectl apply -f  $i; done

#将ip、reids密码、minio账号密码等改成.env文件中的
kubectl patch Secret juicefs-sc-secret -n kube-system -p '{"stringData":{"metaurl":"'"redis://:${REDIS_PASSWORD}@${JUICEFS_HOST_IP}:6382/1"'"}}'
kubectl patch Secret juicefs-sc-secret -n kube-system -p '{"stringData":{"bucket":"'"http://${JUICEFS_HOST_IP}:9010/juicefs"'"}}'
kubectl patch Secret juicefs-sc-secret -n kube-system -p '{"stringData":{"access-key":"'"${MINIO_ROOT_USER}"'"}}'
kubectl patch Secret juicefs-sc-secret -n kube-system -p '{"stringData":{"secret-key":"'"${MINIO_ROOT_PASSWORD}"'"}}'

#生成cube-studio所需的pv及pvc
for i in $(ls cube-pv-pvc-with-juicefs/); do kubectl apply -f  $i; done

#挂载到宿主机的/data/jfs目录，并指定redis的数据库1为元数据存储；这样方便调整、查看service pv、pipline pv中的内容
juicefs mount -d "redis://:${REDIS_PASSWORD}@${JUICEFS_HOST_IP}:6382/1" /data/jfs
#卸载目录
#juicefs umount -d /data/jfs