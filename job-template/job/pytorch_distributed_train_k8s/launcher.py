
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
HEADER_NAME = os.getenv('RAY_HOST', '')
WORKER_NAME = HEADER_NAME.replace('header', 'worker')
INIT_FILE=''
crd_info={
    "group": "kubeflow.org",
    "version": "v1",
    'kind': 'PyTorchJob',
    "plural": "pytorchjobs",
    "timeout": 60 * 60 * 24 * 2
}


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)

GPU_TYPE= os.getenv('KFJ_GPU_TYPE', 'NVIDIA')
GPU_RESOURCE= os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
print(GPU_TYPE,GPU_RESOURCE)



def default_job_name():
    # import re
    # ctx = KFJobContext.get_context()
    # p_name = str(ctx.pipeline_name) or ''
    # p_name = re.sub(r'[^-a-z0-9]', '-', p_name)
    # return "-".join([str(ctx.creator), p_name, "pytorchjob", str(uuid.uuid1())])
    name = "pytorchjob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
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






# 监控指定名称的pytorchjob
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
        pytorchjob = crd_k8s.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,name=name)
        if pytorchjob:
            print('pytorchjob status %s'%pytorchjob['status'], flush=True)
        else:
            print('pytorchjob not exist', flush=True)
        if pytorchjob and (pytorchjob['status']=="Succeeded" or pytorchjob['status']=="Failed"):    # Created, Running, Restarting, Succeeded, or Failed
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            break
        if (datetime.datetime.now()-check_time).seconds>3600:
            pids = get_pid("stern")
            if pids:
                for pid in pids:
                    pro = psutil.Process(int(pid))
                    pro.terminate()
                    print('kill process %s'%pid, flush=True)
            check_time=datetime.datetime.now()
        time.sleep(60)



# @pysnooper.snoop()
def make_pytorchjob(name,num_workers,image,working_dir,command):
    # if type(command)==str:
    #     command=command.split(" ")
    #     command = [c for c in command if c]
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
                    "type": "pytorchjob",
                    "run-id": KFJ_RUN_ID,
                }
            },
            "spec": {
                "schedulerName": "kube-batch",
                "restartPolicy": "Never",
                "volumes": k8s_volumes,
                # "imagePullSecrets": [
                #     {
                #         "name": "hubsecret"
                #     }
                # ],
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
                                            "type": "pytorchjob"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "containers": [
                    {
                        "name": "pytorch",
                        "image": image if image else KFJ_TASK_IMAGES,
                        "imagePullPolicy": "Always",
                        "workingDir":working_dir,
                        "env":[
                            {
                                "name": "NCCL_DEBUG",
                                "value":"INFO"
                            },
                            {
                                "name": "NCCL_IB_DISABLE",
                                "value": "1"
                            },
                            # {
                            #     "name": "NCCL_DEBUG_SUBSYS",
                            #     "value": "ALL"
                            # },
                            {
                                "name": "NCCL_SOCKET_IFNAME",
                                "value": "eth0"
                            }
                        ],
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
    worker_pod_spec['replicas']=int(num_workers)-1   # 因为master是其中一个worker

    pytorch_deploy = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
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
            "backoffLimit":num_workers,
            "cleanPodPolicy": "None",
            "pytorchReplicaSpecs": {
                "Master":pod_spec,
                "Worker":worker_pod_spec
            }

        }
    }

    return pytorch_deploy


# @pysnooper.snoop()
def launch_pytorchjob(name, num_workers, image,working_dir, worker_command):
    """
    由给定参数启动PytorchJob进行模型训练
    Args:
        name: TFJob任务名字，如果不传，默认为"pytorchjob_<uuid>"
        namespace: pytorchJob的namespace，如果不传，默认为"kubeflow"
        num_workers: 训练使用的机器数
        driver_cmd: 训练镜像的入口命令
        driver_image: 训练镜像地址
        driver_args: 训练脚本启动参数
        driver_envs: 环境变量
        driver_pvc_name: 训练docker挂载的pvc的名字
        driver_pvc_mount_path: pvc挂载到训练docker中的路径
        node_select: 节点选择
        job_timeout: 训练任务的最大运行时间，
    Returns:
    """

    if KFJ_RUN_ID:
        print('delete old pytorch, run-id %s'%KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        time.sleep(10)
    # 删除旧的pytorch
    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)
    # 创建新的pytorch
    pytorchjob_json = make_pytorchjob(name=name,num_workers= num_workers,image = image,working_dir=working_dir,command=worker_command)
    print('create new pytorch %s' % name, flush=True)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=pytorchjob_json)
    time.sleep(10)

    print('begin start monitoring thred', flush=True)
    # # 后台启动监控脚本
    monitoring_thred = threading.Thread(target=monitoring,args=(k8s_client,name,KFJ_NAMESPACE))
    monitoring_thred.start()
    while True:
        # 实时打印日志
        line='>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print('begin follow log\n%s'%line, flush=True)
        command = "stern %s --namespace %s --exclude-container init-pytorch --tail 10 --template '{{.PodName}} {{.Message}}'"%(name,KFJ_NAMESPACE)
        print(command, flush=True)
        run_shell(command)
        print('%s\nend follow log'%line, flush=True)
        time.sleep(10)

        pytorchjob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        if pytorchjob and (pytorchjob['status'] == "Succeeded" or pytorchjob['status'] == "Failed"):
            break

    pytorchjob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
    print("pytorchJob %s finished, status %s"%(name, pytorchjob['status']))

    if pytorchjob['status']!='Succeeded':
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("TFjob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录", default='/mnt/')
    arg_parser.add_argument('--command', type=str, help="运行job的命令", default='python3 mnist.py')
    arg_parser.add_argument('--num_worker', type=int, help="运行job所在的机器", default=3)
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='')

    args = arg_parser.parse_args()
    print("{} args: {}".format(__file__, args))


    launch_pytorchjob(name=default_job_name(),num_workers=args.num_worker,image=args.image,working_dir=args.working_dir,worker_command=args.command)


