
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
KFJ_TASK_PROJECT_NAME = os.getenv('KFJ_TASK_PROJECT_NAME', 'public')
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
    "group": "batch.volcano.sh",
    "version": "v1alpha1",
    'kind': 'Job',
    "plural": "jobs",
    "timeout": 60 * 60 * 24 * 2
}


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)

GPU_RESOURCE_NAME= os.getenv('GPU_RESOURCE_NAME', '')
GPU_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
gpu_num,gpu_type,_ = k8s_client.get_gpu(GPU_RESOURCE)
if gpu_type:
    KFJ_TASK_NODE_SELECTOR['gpu-type']=gpu_type

RDMA_RESOURCE_NAME= os.getenv('RDMA_RESOURCE_NAME', '')
RDMA_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_RDMA', '0')

HUBSECRET = os.getenv('HUBSECRET','hubsecret')
HUBSECRET=[{"name":hubsecret} for hubsecret in HUBSECRET.split(',')]

DEFAULT_POD_RESOURCES = os.getenv('DEFAULT_POD_RESOURCES','')
DEFAULT_POD_RESOURCES = json.loads(DEFAULT_POD_RESOURCES) if DEFAULT_POD_RESOURCES else {}

def default_job_name():
    name = "volcanojob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
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
        if status==-9 or status==-15 or status==143 or status==127:   # 外界触发kill
        # if status!=0:  # 外界触发kill
            print('shell finish %s'%status,flush=True)
            break

    return cmd.returncode






# 监控指定名称的volcanojob
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
        volcanojob = crd_k8s.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=name)
        status = volcanojob['status'].lower()
        if volcanojob:
            print('volcanojob status %s'%volcanojob['status'], flush=True)
        else:
            print('volcanojob not exist', flush=True)

        if volcanojob and (status=="completed" or status=="failed" or status=='aborted' or status=='terminated'):    # Created, Running, Restarting, Completed, or Failed
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
def make_volcanojob(name,num_workers,image,working_dir,command):
    # if type(command)==str:
    #     command=command.split(" ")
    #     command = [c for c in command if c]
    task_spec={
        "replicas": num_workers,
        "name": "worker",
        "template": {
            "metadata": {
                "labels": {
                    "pipeline-id": KFJ_PIPELINE_ID,
                    "pipeline-name": KFJ_PIPELINE_NAME,
                    "task-id": KFJ_TASK_ID,
                    "task-name": KFJ_TASK_NAME,
                    'rtx-user': KFJ_RUNNER,
                    "component": name,
                    "type": "volcanojob",
                    "run-id": KFJ_RUN_ID,
                },
                "annotations": {
                    "project": KFJ_TASK_PROJECT_NAME
                }
            },
            "spec": {
                "restartPolicy": "Never",
                "volumes": k8s_volumes,
                "imagePullSecrets": HUBSECRET,
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
                                "weight": 20,
                                "podAffinityTerm": {
                                    "topologyKey": "kubernetes.io/hostname",
                                    "labelSelector": {
                                        "matchLabels": {
                                            "component": name,
                                            "type": "volcanojob"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "containers": [
                    {
                        "name": "volcanojob",
                        "image": image if image else KFJ_TASK_IMAGES,
                        "imagePullPolicy": "Always",
                        "workingDir":working_dir,
                        "env":[
                            {
                                "name": "NCCL_DEBUG",
                                "value":"INFO"
                            }
                        ],
                        "command": ['bash','-c',command],
                        "volumeMounts": k8s_volume_mounts,
                        "resources": {
                            "requests": {
                                **{
                                    "cpu": KFJ_TASK_RESOURCE_CPU,
                                    "memory": KFJ_TASK_RESOURCE_MEMORY,
                                },
                                **DEFAULT_POD_RESOURCES
                            },
                            "limits": {
                                **{
                                    "cpu": KFJ_TASK_RESOURCE_CPU,
                                    "memory": KFJ_TASK_RESOURCE_MEMORY
                                },
                                **DEFAULT_POD_RESOURCES
                            }
                        }
                    }
                ]
            }
        }
    }


    if int(gpu_num):
        task_spec['template']['spec']['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
        task_spec['template']['spec']['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
    else:
        # 添加禁用指令
        task_spec['template']['spec']['containers'][0]['env'].append({
            "name":"NVIDIA_VISIBLE_DEVICES",
            "value":"none"
        })
    worker_pod_spec = copy.deepcopy(task_spec)
    worker_pod_spec['replicas']=int(num_workers)-1   # 因为master是其中一个worker

    volcano_deploy = {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
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
            },
            "annotations": {
                "project": KFJ_TASK_PROJECT_NAME
            }
        },
        "spec": {
            "minAvailable":num_workers,
            "schedulerName":"volcano",
            "cleanPodPolicy": "None",
            "plugins":{
                "env": [],
                "svc": [],
                "ssh": []
            },
            "queue":"default",
            "tasks": [
                task_spec
            ]
        }
    }

    return volcano_deploy


# @pysnooper.snoop()
def launch_volcanojob(name, num_workers, image,working_dir, worker_command):
    if KFJ_RUN_ID:
        print('delete old volcanojob, run-id %s'%KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        time.sleep(10)
    # 删除旧的volcanojob
    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)
    # 创建新的volcanojob
    volcanojob_json = make_volcanojob(name=name,num_workers= num_workers,image = image,working_dir=working_dir,command=worker_command)
    print(volcanojob_json)
    print('create new volcanojob %s' % name, flush=True)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=volcanojob_json)
    time.sleep(10)

    print('begin start monitoring thread', flush=True)
    # # 后台启动监控脚本
    monitoring_thread = threading.Thread(target=monitoring,args=(k8s_client,name,KFJ_NAMESPACE))
    monitoring_thread.start()
    while True:
        # 实时打印日志,由于stern跟踪时长有限制，所以会每个小时重启一次ster进程
        line='>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print('begin follow log\n%s %s'%(line,datetime.datetime.now()), flush=True)
        command = '''stern %s --namespace %s --since 10s --template '{{.PodName}} {{.Message}} {{"\\n"}}' '''%(name,KFJ_NAMESPACE)
        print(command, flush=True)
        run_shell(command)
        print('%s %s\nend follow log'%(line,datetime.datetime.now()), flush=True)
        time.sleep(10)

        volcanojob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        if volcanojob and (volcanojob['status'] == "Completed" or volcanojob['status'] == "Failed"):
            break

    volcanojob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
    print("volcanojob %s finished, status %s"%(name, volcanojob['status']))

    if volcanojob['status']!='Completed':
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("volcanojob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录", default='/mnt/')
    arg_parser.add_argument('--command', type=str, help="运行job的命令", default='python3 mnist.py')
    arg_parser.add_argument('--num_worker', type=int, help="分布式worker的数量", default=3)
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='')

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))

    launch_volcanojob(name=default_job_name(),num_workers=args.num_worker,image=args.image,working_dir=args.working_dir,worker_command=args.command)


