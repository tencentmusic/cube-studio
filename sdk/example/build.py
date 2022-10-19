
print('docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:base  -f Dockerfile .')
print('docker push ccr.ccs.tencentyun.com/cube-studio/aihub:base')
import os,sys,time,json,shutil
g = os.walk("./")
for path,dir_list,file_list in g:
    for app_name in dir_list:
        if 'app1' not in app_name:
            dockerfile_path = os.path.join(path, app_name,'Dockerfile')
            if os.path.exists(dockerfile_path):
                command = "cd %s && docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:%s . && docker push ccr.ccs.tencentyun.com/cube-studio/aihub:%s && cd ../"%(app_name,app_name,app_name)
                print(command)



