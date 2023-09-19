
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)

import argparse
import datetime
import json
import time
import uuid
import os
import pysnooper
import os,sys
import re
import threading
import psutil
import copy

from kubernetes import client

# print(os.environ)
from job.pkgs.k8s.py_k8s import K8s
k8s_client = K8s()

KFJ_NAMESPACE = os.getenv('KFJ_NAMESPACE', '')
KFJ_TASK_ID = os.getenv('KFJ_TASK_ID', '')
KFJ_TASK_NAME = os.getenv('KFJ_TASK_NAME', '')
task_node_selectors = re.split(',|;|\n|\t', os.getenv('KFJ_TASK_NODE_SELECTOR', 'cpu=true,train=true'))
KFJ_TASK_NODE_SELECTOR = {}
for task_node_selector in task_node_selectors:
    KFJ_TASK_NODE_SELECTOR[task_node_selector.split('=')[0]] = task_node_selector.split('=')[1]

KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_CREATOR = os.getenv('KFJ_CREATOR', '')
KFJ_RUNNER = os.getenv('KFJ_RUNNER','')
KFJ_PIPELINE_NAME = os.getenv('KFJ_PIPELINE_NAME', '')
KFJ_TASK_IMAGES = os.getenv('KFJ_TASK_IMAGES', '')
KFJ_TASK_VOLUME_MOUNT = os.getenv('KFJ_TASK_VOLUME_MOUNT', '')
KFJ_TASK_RESOURCE_CPU = os.getenv('KFJ_TASK_RESOURCE_CPU', '')
KFJ_TASK_RESOURCE_MEMORY = os.getenv('KFJ_TASK_RESOURCE_MEMORY', '')
NUM_WORKER = 3
INIT_FILE=''
crd_info={
    "group": "kubeflow.org",
    "version": "v1",
    'kind': 'PaddleJob',
    "plural": "paddlejobs",
    "timeout": 60 * 60 * 24 * 2
}


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)

GPU_TYPE= os.getenv('KFJ_GPU_TYPE', 'NVIDIA')
GPU_RESOURCE= os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
print(GPU_TYPE,GPU_RESOURCE)
schedulerName = os.getenv('SCHEDULER', 'default-scheduler')



def default_job_name():
    name = "paddlejob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
    return name[0:54]




import subprocess
# @pysnooper.snoop()
def run_shell(shell):
    print('begin run shell: %s'%shell,flush=True)
    cmd = subprocess.Popen(shell, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE, universal_newlines=True, shell=True, bufsize=1)
    # 实时输出
    while True:
        line = cmd.stdout.readline()
        status = subprocess.Popen.poll(cmd)
        if status:
            print(status,line,end='', flush=True)
        else:
            print(line, end='', flush=True)
        if status == 0:  # 判断子进程是否结束
            print('shell finish %s'%status,flush=True)
            break
        if status==-9 or status==-15 or status==143:   # 外界触发kill
            print('shell finish %s'%status,flush=True)
            break

    return cmd.returncode






# 监控指定名称的paddlejob
def monitoring(crd_k8s,name,namespace):
    time.sleep(10)
    # 杀掉stern 进程
    def get_pid(name):
        '''
         作用：根据进程名获取进程pid
        '''
        pids = psutil.process_iter()
        print("[" + name + "]'s pid is:", flush=True)
        back=[]
        for pid in pids:
            if name in pid.name():
                print(pid.pid, flush=True)
                back.append(pid.pid)
        return back
    check_time = datetime.datetime.now()
    while(True):
        paddlejob = crd_k8s.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=name)
        if paddlejob:
            print('paddlejob status %s'%paddlejob['status'], flush=True)
        else:
            print('paddlejob not exist', flush=True)
        if paddlejob and (paddlejob['status']=="Succeeded" or paddlejob['status']=="Failed" or paddlejob['status']=='Completed'):    # Created, Running, Restarting, Succeeded, or Failed
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            break
        if (datetime.datetime.now()-check_time).total_seconds()>3600:
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            check_time=datetime.datetime.now()
        time.sleep(60)



# @pysnooper.snoop()
def make_paddlejob(name,num_workers,num_ps,image,working_dir,command):
    pod_spec={
        "replicas": 1,
        "restartPolicy": "Never",
        "template": {
            "metadata": {
                "labels": {
                    "pipeline-id": KFJ_PIPELINE_ID,
                    "pipeline-name": KFJ_PIPELINE_NAME,
                    "task-id": KFJ_TASK_ID,
                    "task-name": KFJ_TASK_NAME,
                    'rtx-user': KFJ_RUNNER,
                    "component": name,
                    "type": "paddlejob",
                    "run-id": KFJ_RUN_ID,
                }
            },
            "spec": {
                "schedulerName": schedulerName,
                "restartPolicy": "Never",
                "volumes": k8s_volumes,
                "imagePullSecrets": [
                    {
                        "name": "hubsecret"
                    }
                ],
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": node_selector_key,
                                            "operator": "In",
                                            "values": [
                                                KFJ_TASK_NODE_SELECTOR[node_selector_key]
                                            ]
                                        } for node_selector_key in KFJ_TASK_NODE_SELECTOR
                                    ]
                                }
                            ]
                        }
                    },
                    "podAntiAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "weight": 5,
                                "podAffinityTerm": {
                                    "topologyKey": "kubernetes.io/hostname",
                                    "labelSelector": {
                                        "matchLabels": {
                                            "component": name,
                                            "type": "paddlejob"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "containers": [
                    {
                        "name": "paddle",
                        "image": image if image else KFJ_TASK_IMAGES,
                        "imagePullPolicy": "Always",
                        "workingDir":working_dir,
                        "command": ['bash','-c',command],
                        "volumeMounts": k8s_volume_mounts,
                        "resources": {
                            "requests": {
                                "cpu": KFJ_TASK_RESOURCE_CPU,
                                "memory": KFJ_TASK_RESOURCE_MEMORY,
                            },
                            "limits": {
                                "cpu": KFJ_TASK_RESOURCE_CPU,
                                "memory": KFJ_TASK_RESOURCE_MEMORY
                            }
                        }
                    }
                ]
            }
        }
    }


    if GPU_TYPE=='NVIDIA' and GPU_RESOURCE:
        pod_spec['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]
        pod_spec['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]

    worker_pod_spec = copy.deepcopy(pod_spec)
    worker_pod_spec['replicas']=int(num_workers)
    ps_pod_spec = copy.deepcopy(pod_spec)
    ps_pod_spec['replicas']=int(num_ps)
    paddle_deploy = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PaddleJob",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels":{
                "run-id":KFJ_RUN_ID,
                "run-rtx":KFJ_RUNNER,
                "pipeline-rtx": KFJ_CREATOR,
                "pipeline-id": KFJ_PIPELINE_ID,
                "pipeline-name": KFJ_PIPELINE_NAME,
                "task-id": KFJ_TASK_ID,
                "task-name": KFJ_TASK_NAME,
            }
        },
        "spec": {
            # "withGloo":1, # withGloo的可选配置为0不启用，1只启动worker端，2启动所有（worker和server），建议设置1；
            "intranet":'PodIP',
            "cleanPodPolicy": "Never",
            "worker":worker_pod_spec
        }
    }
    if num_ps:
        paddle_deploy['spec']['ps']=ps_pod_spec

    return paddle_deploy


# @pysnooper.snoop()
def launch_paddlejob(name, num_workers,num_ps, image,working_dir, worker_command):

    if KFJ_RUN_ID:
        print('delete old paddle, run-id %s'%KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        time.sleep(10)
    # 删除旧的paddle
    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)
    # 创建新的paddle
    paddlejob_json = make_paddlejob(name=name,num_workers= num_workers,num_ps=num_ps, image = image,working_dir=working_dir,command=worker_command)
    print('create new paddle %s' % name, flush=True)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=paddlejob_json)
    time.sleep(10)

    print('begin start monitoring thread', flush=True)
    # # 后台启动监控脚本
    monitoring_thread = threading.Thread(target=monitoring,args=(k8s_client,name,KFJ_NAMESPACE))
    monitoring_thread.start()
    while True:
        # 实时打印日志
        line='>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print('begin follow log\n%s'%line, flush=True)
        command = '''stern %s --namespace %s --exclude-container coord-paddle --since 10s --template '{{.PodName}} {{.Message}} {{"\\n"}}' '''%(name,KFJ_NAMESPACE)

        print(command, flush=True)
        run_shell(command)
        print('%s\nend follow log'%line, flush=True)
        time.sleep(10)

        paddlejob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        if paddlejob and (paddlejob['status'] == "Succeeded" or paddlejob['status'] == "Failed" or paddlejob['status'] == "Completed"):
            break

    paddlejob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
    print("paddleJob %s finished, status %s"%(name, paddlejob['status']))

    if paddlejob['status']!='Succeeded' and paddlejob['status']!="Completed":
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Paddlejob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录", default='/mnt/')
    arg_parser.add_argument('--command', type=str, help="运行job的命令", default='python3 mnist.py')
    arg_parser.add_argument('--num_ps', type=int, help="运行ps的pod数目", default=0)
    arg_parser.add_argument('--num_worker', type=int, help="运行worker的pod数目", default=3)
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='')

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))


    launch_paddlejob(name=default_job_name(),num_workers=int(args.num_worker),num_ps=int(args.num_ps),image=args.image,working_dir=args.working_dir,worker_command=args.command)


