

import os,sys,time,json,shutil
path = os.path.dirname(os.path.abspath(__file__))


# 生成构建镜像的脚本
# build_file=open('build.sh',mode='w')
# for app_name in os.listdir("."):
#     if os.path.isdir(app_name):
#         if app_name in ['__pycache__','deploy','app1']:
#             continue
#         dockerfile_path = os.path.join(app_name,'Dockerfile')
#         app_name = app_name.lower().replace('_', '-')
#         if os.path.exists(dockerfile_path):
#             command = f"docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:{app_name} ./{app_name}/ && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:{app_name} &"
#             build_file.write(command)
#             build_file.write('\n')
#
# build_file.write('\n\nwait')
# build_file.close()


# 生成部署脚本和info
all_info=[]
env='cloud'     # env='dev'
for app_name in os.listdir("."):
    if os.path.isdir(app_name):
        if app_name in ['__pycache__','deploy']:
            continue
        # 内网部署必须要info.json
        info={
            "name":app_name
        }
        dockerfile_path = os.path.join(app_name, 'Dockerfile')
        if not os.path.exists(dockerfile_path):
            continue

        if env=='dev':
            if not os.path.exists(os.path.join(app_name,'info.json')):
                continue

            info = json.load(open(os.path.join(app_name,'info.json')))
            if info.get('status','offline')=='offline':
                continue

        app_name = app_name.lower().replace('_', '-')
        if app_name != 'app1':
            all_info.append(info)
        if env=='cloud':
            synchronous = 'asynchronous'
            resource_gpu = '0'
            info['doc'] = f"http://www.data-master.net:8888/aihub/{app_name}"
            host='www.data-master.net'
        else:
            synchronous = 'synchronous'
            resource_gpu = info.get('resource_gpu','0')
            info['doc'] = f"http://aihub.cube.woa.com/aihub/{app_name}"
            host='aihub.cube.woa.com'

        # 生成k8s部署的脚本
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
            path: /data/k8s/kubeflow/pipeline/workspace/pengluan/cube-studio/aihub/deep-learning/{app_name}
        - name: cube-studio
          hostPath:
            path: /data/k8s/kubeflow/pipeline/workspace/pengluan/cube-studio/aihub/src
        - name: cos-data
          hostPath:
            path: /mnt/aihub
      nodeSelector:
        aihub: {'cpu' if resource_gpu=='0' else 'gpu'}
      containers:
        - name: aihub-{app_name}
          image: ccr.ccs.tencentyun.com/cube-studio/aihub:{app_name}
          imagePullPolicy: Always  # IfNotPresent
          command: ["bash", "-c", "/src/docker/entrypoint.sh python app.py"]
          securityContext:
            privileged: true
          env:
          - name: APPNAME
            value: {app_name}
          - name: REDIS_URL
            value: redis://:admin@43.142.20.178:6379/0
          - name: REQ_TYPE
            value: {synchronous}
          - name: NVIDIA_VISIBLE_DEVICES
            value: all
          volumeMounts:
            - name: tz-config
              mountPath: /etc/localtime
            - name: app-data
              mountPath: /app
            - name: cube-studio
              mountPath: /src
            - name: cos-data
              mountPath: /src/cubestudio/aihub/web/static

---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: aihub-{app_name}
  namespace: aihub
spec:
  gateways:
  - kubeflow/kubeflow-gateway-8080
  hosts:
  - "{host}"
  http:
  - match:
    - uri:
        prefix: /{'aihub/' if app_name=='app1' else app_name+"/"}
    route:
    - destination:
        host: aihub-{app_name}.aihub.svc.cluster.local
        port:
          number: 80
        '''
        os.makedirs('deploy',exist_ok=True)
        save_path = f"deploy/{app_name}.yaml"
        # print(save_path)
        file = open(save_path,mode='w')
        file.write(deploy)
        file.close()

file = open('info.json',mode='w')
file.write(json.dumps(all_info,indent=2,ensure_ascii=False))
file.close()