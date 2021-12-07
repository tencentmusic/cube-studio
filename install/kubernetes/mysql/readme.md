# k8s部署mysql 

## 为机器打label 
kubectl label node node1xx mysql=true --overwrite

## 创建pv，pvc，根据自己的实际情况创建(内置的账号密码为root/admin)
kubectl create -f pv-pvc-hostpath.yaml   
kubectl create -f service.yaml     
kubectl create -f configmap-mysql.yaml   
kubectl create -f deploy.yaml  

## 校验mysql的pv和pvc是否匹配完成

# 本地调试可以使用docker启动mysql
docker run -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=admin -d mysql:5.7  
