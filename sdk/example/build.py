

from cubestudio.util.py_shell import exec
import os,sys,time,json,shutil
g = os.walk("./")
for path,dir_list,file_list in g:
    for dir_name in dir_list:
        dockerfile_path = os.path.join(path, dir_name,'Dockerfile')
        if os.path.exists(dockerfile_path):
            command = "cd %s && docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:%s ."%(dir_name,dir_name)
            print(command)
            # exec(command)



