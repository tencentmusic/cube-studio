
#   生成构建镜像的脚本

import os,sys,time,json,shutil
path = os.path.dirname(os.path.abspath(__file__))
all_info=[]
for app_name in os.listdir("."):
    if os.path.isdir(app_name):
        if app_name in ['__pycache__','app1','deploy']:
            continue
        if not os.path.exists(os.path.join(app_name,'info.json')):
            continue

        info = json.load(open(os.path.join(app_name,'info.json')))
        if info.get('status','offline')=='offline':
            continue


        app_name = app_name.lower().replace('_', '-')
        info['doc']=f"http://{app_name}.aihub.cube.woa.com/frontend/{app_name}"
        all_info.append(info)
        resource_gpu=info.get('inference',{}).get('resource_gpu','0')
        if 'http' not in info['pic']:
            info['pic']=f"http://{app_name}.aihub.cube.woa.com/{app_name}/static/example/"+app_name+"/" + info['pic']


        # 批量构建镜像
        dockerfile_path = os.path.join(app_name,'Dockerfile')
        if os.path.exists(dockerfile_path):
            command = "docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:%s ./%s/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:%s &"%(app_name,app_name,app_name)
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
    - name: backend
      port: 8080
      targetPort: 8080
      protocol: TCP
    - name: frontend
      port: 80
      targetPort: 80
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
        - name: app-data
          hostPath:
            path: /data/k8s/kubeflow/pipeline/workspace/pengluan/cube-studio/sdk/example/{app_name}
        - name: cube-studio
          hostPath:
            path: /data/k8s/kubeflow/pipeline/workspace/pengluan/cube-studio/sdk/src
      nodeSelector:
        aihub: {'cpu' if resource_gpu=='0' else 'gpu'}
      containers:
        - name: aihub-{app_name}
          image: ccr.ccs.tencentyun.com/cube-studio/aihub:{app_name}
          imagePullPolicy: Always  # IfNotPresent
          command: ["bash","-c","pip install celery redis && bash /entrypoint.sh"]
          securityContext:
            privileged: true
          env:
          - name: APPNAME
            value: {app_name}
          - name: REDIS_URL
            value: redis://:admin@43.142.20.178:6379/0
          - name: REQ_TYPE
            value: synchronous
          - name: NVIDIA_VISIBLE_DEVICES
            value: all
          volumeMounts:
            - name: tz-config
              mountPath: /etc/localtime
            - name: app-data
              mountPath: /app
            - name: cube-studio
              mountPath: /src
              
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: aihub-{app_name}
  namespace: aihub
spec:
  gateways:
  - kubeflow/kubeflow-gateway
  hosts:
  - "{app_name}.aihub.cube.woa.com"  
  http:
  - route:
    - destination:
        host: aihub-{app_name}.aihub.svc.cluster.local
        port:
          number: 80
        '''
        save_path = f"deploy/{app_name}.yaml"
        # print(save_path)
        file = open(save_path,mode='w')
        file.write(deploy)
        file.close()

print('\n\nwait')

file = open('info.json',mode='w')
file.write(json.dumps(all_info,indent=2,ensure_ascii=False))
file.close()