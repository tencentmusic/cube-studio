# -*- coding: utf-8 -*-
import os,sys
base_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_dir)
sys.path.append(os.path.realpath(__file__))

from common import logging, nonBlockRead, HiddenPrints

import argparse
import datetime
import json
import time
import uuid
import pysnooper
import re
import subprocess
import psutil
import copy

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

crd_info={
    "group": "batch.volcano.sh",
    "version": "v1alpha1",
    'kind': 'Job',
    "plural": "jobs",
    "timeout": 60 * 60 * 24 * 2
}
k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

GPU_TYPE = os.getenv('KFJ_GPU_TYPE', 'NVIDIA')
GPU_RESOURCE = os.getenv('KFJ_TASK_RESOURCE_GPU', '0')

def default_job_name():
    name = "kaldi-volcanojob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
    return name[0:54]

# @pysnooper.snoop()
def make_volcanojob(name, num_workers, image, working_dir, worker_command, master_command):
    worker_task_spec={
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
                }
            },
            "spec": {
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
                            },
                            {
                                "name": "MY_POD_IP",
                                "valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "status.podIP"}}
                            },
                            {
                                "name": "KFJ_RUN_ID",
                                "value": str(KFJ_RUN_ID),
                            },
                            {
                                "name": "TZ",
                                "value": "Asia/Shanghai"
                            }
                        ],
                        "command": ['bash','-c',worker_command],
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
                        },
                        "securityContext": {"runAsUser": 0}
                    }
                ]
            }
        }
    }

    master_task_spec = copy.deepcopy(worker_task_spec)
    master_task_spec['replicas'] = 1
    master_task_spec['name'] = 'master'
    master_task_spec['template']['spec']['containers'][0]['command'] = ['bash','-c',master_command]
    master_task_spec['policies'] = [{"event": "TaskCompleted", "action": "CompleteJob"}, {"event": "PodFailed", "action": "AbortJob"}]


    if GPU_TYPE=='NVIDIA' and GPU_RESOURCE:
        worker_task_spec['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]
        worker_task_spec['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]

    if GPU_TYPE=='TENCENT' and GPU_RESOURCE:
        if len(GPU_RESOURCE.split(','))==2:
            gpu_core,gpu_mem = GPU_RESOURCE.split(',')[0],str(4*int(GPU_RESOURCE.split(',')[1]))
            if gpu_core and gpu_mem:
                worker_task_spec['template']['spec']['containers'][0]['resources']['requests']['tencent.com/vcuda-core'] = gpu_core
                worker_task_spec['template']['spec']['containers'][0]['resources']['requests']['tencent.com/vcuda-memory'] = gpu_mem
                worker_task_spec['template']['spec']['containers'][0]['resources']['limits']['tencent.com/vcuda-core'] = gpu_core
                worker_task_spec['template']['spec']['containers'][0]['resources']['limits']['tencent.com/vcuda-memory'] = gpu_mem

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
            }
        },
        "spec": {
            "minAvailable":num_workers + 1,
            "schedulerName":"volcano",
            "cleanPodPolicy": "None",
            "plugins":{
                "ssh":[],
                "env":[],
                "svc":[]
            },
            "queue":"default",
            "tasks": [
                worker_task_spec,
                master_task_spec
            ]
        }
    }
    return volcano_deploy

# @pysnooper.snoop()
def launch_volcanojob(name, num_workers, image, working_dir, worker_command, master_command):
    # 删除旧的volcanojob
    if KFJ_RUN_ID:
        logging.info('Try delete old volcanojob.')
    with HiddenPrints():
        try:
            k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
        except:
            logging.info('Nothing to delete, name:{}, run_id:{}'.format(name, KFJ_RUN_ID))
    time.sleep(10)

    # 创建新的volcanojob
    volcanojob_json = make_volcanojob(name=name,num_workers= num_workers,image = image,working_dir=working_dir,worker_command=worker_command, master_command=master_command)

    logging.info('Create new volcanojob %s' % name)
    logging.info(volcanojob_json)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=volcanojob_json)

		# 开子程程跟踪日志
    logging.info('Begin tracing logs')
    command = "sh stern.sh '{name}' '{namespace}'".format(name=name, namespace=KFJ_NAMESPACE)
    log_proc = subprocess.Popen(command,stdin=subprocess.PIPE, stderr=subprocess.PIPE, \
                            stdout=subprocess.PIPE, universal_newlines=True, shell=True, bufsize=1)

		# 阻塞到Volcanojob结束或者用户任务结束
    loop_time = 0
    while True:
        time.sleep(10)
        volcanojob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)
        if not volcanojob: # 获取volcanojob失败，再给一次机会
        		time.sleep(10)
        		volcanojob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)

        line = nonBlockRead(log_proc.stdout)
        while line:
            logging.info(line.strip())
            line = nonBlockRead(log_proc.stdout)

        if loop_time % 60 == 0:
            logging.info("heartbeating, volcanojob status: {}".format(str(volcanojob['status'])))
        loop_time += 1

        if not volcanojob:
            break
        if volcanojob and (volcanojob['status'] not in ("Running","Pending")):
            break

    if not volcanojob:
        raise RuntimeError # ("Volcanojob disappear!!!Check if deleted artificial!!!")
    if volcanojob['status']!='Completed':
        logging.error("volcanojob %s finished, status %s"%(name, volcanojob['status']))
        raise RuntimeError # ("volcanojob %s finished, status %s"%(name, volcanojob['status']))
    logging.info("Volcanojob %s finished, status %s"%(name, volcanojob['status']))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("volcanojob launcher")
    arg_parser.add_argument('--working_dir', type=str, help="运行job的工作目录,需要分布式存储", default='')
    arg_parser.add_argument('--num_worker', type=int, help="workder数量", default=3)
    arg_parser.add_argument('--user_cmd', type=str, help="执行命令", default='./run.sh')
    arg_parser.add_argument('--image', type=str, help="worker的镜像", default='mirrors.tencent.com/raw-kaldi/liutaozhang_kaldi_gpu:latest')

    args = arg_parser.parse_args()
    logging.info("{} args: {}".format(__file__, args))

    if not os.path.exists(args.working_dir):
        raise RuntimeError # (args.working_dir + " not exits!!!")
    # 注入host到启动pod
    if os.system("echo '10.101.140.98 cls-g9v4gmm0.ccs.tencent-cloud.com' >> /etc/hosts"):
        raise RuntimeError # ("Init hosts fail!!!")
    # 创建run_dir到工作目录
    run_dir = "{}/run_{}/".format(args.working_dir, KFJ_RUN_ID)
    if os.system("mkdir -p {}".format(run_dir)):
        raise RuntimeError # create run_dir fail!!!
    # 注入k8s.pl到工作目录
    if os.system("cp /app/*.pl {run_dir}/ && cp /app/*.sh {run_dir}/ && chmod 777 {run_dir}/*".format(run_dir=run_dir)):
        raise RuntimeError # cp k8s.pl fail!!!

    machine_dir = run_dir + '/'+ 'machines'
    log_dir = run_dir + '/'+ 'run.log'
    worker_command = "{run_dir}/worker.sh {run_dir}".format(run_dir=run_dir)
    master_command = "{run_dir}/master.sh {run_dir} {num_worker} \"{user_cmd}\"".format(run_dir=run_dir, num_worker=str(args.num_worker), user_cmd=str(args.user_cmd))
    launch_volcanojob(name=default_job_name(),num_workers=args.num_worker,image=args.image,working_dir=args.working_dir,worker_command=worker_command, master_command=master_command)

