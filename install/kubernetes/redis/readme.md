# 部署redis

## 为机器打label
kubectl label nodes nodexxxx redis=true --overwrite

## 部署pv/pvc根据自己的实际创建(内置的密码为admin)
```bash
kubectl create -f pv-hostpath.yaml  
kubectl create -f configmap.yaml
kubectl create -f service.yaml
# 如果自己需要使用pv来保存redis队列数据，可以修改master.yaml
kubectl create -f master.yaml  
```





