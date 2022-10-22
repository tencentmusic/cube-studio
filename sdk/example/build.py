
#   生成构建镜像的脚本

print('docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base  -f Dockerfile .')
print('docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base')
import os,sys,time,json,shutil
for app_name in os.listdir("."):
    if os.path.isdir(app_name):
        if app_name in ['__pycache__']:
            continue

        app_name=app_name.lower()

        # 批量构建镜像
        dockerfile_path = os.path.join(app_name,'Dockerfile')
        if os.path.exists(dockerfile_path):
            command = "cd %s && docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:%s . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:%s && cd ../"%(app_name,app_name,app_name)
            print(command)

        # 生成部署的脚本
        deploy=f'''
apiVersion: v1
kind: Service
metadata:
  name: aihub-{app_name}
  namespace: aihub
  labels:
    app: aihub-{app_name}
spec:
  ports:
    - name: http
      port: 8080
      targetPort: 8080
      protocol: TCP
  selector:
    app: aihub-{app_name}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aihub-{app_name}
  namespace: aihub
  labels:
    app: aihub-{app_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aihub-{app_name}
  template:
    metadata:
      name: aihub-{app_name}
      labels:
        app: aihub-{app_name}
    spec:
      volumes:
        - name: tz-config
          hostPath:
            path: /usr/share/zoneinfo/Asia/Shanghai
      containers:
        - name: aihub-{app_name}
          image: ccr.ccs.tencentyun.com/cube-studio/aihub:{app_name}
          imagePullPolicy: Always  # IfNotPresent
          env:
          - name: APP_NAME
            value: {app_name}
          volumeMounts:
            - name: tz-config
              mountPath: /etc/localtime
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          resources:
            requests:
              cpu: 0
              memory: 0Gi
        '''
        save_path = app_name+"/deploy.yaml"
        # print(save_path)
        file = open(save_path,mode='w')
        file.write(deploy)
        file.close()

