
import ray
import re

import os
import sys
import time

from job.pkgs.k8s.py_k8s import K8s
k8s_client = K8s()

import argparse
import datetime, time
import pysnooper

# print(os.environ)
base_dir = os.path.split(os.path.realpath(__file__))[0]
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
workingDir='/'


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

def create_header_service(name):
    service_json = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels":{
                "run-id":os.getenv('KFJ_RUN_ID','unknown'),
                "run-rtx":os.getenv('KFJ_RUNNER','unknown'),
                "pipeline-rtx": os.getenv('KFJ_CREATOR', 'unknown'),
                "task-id":os.getenv('KFJ_TASK_ID','unknown'),
                "pipeline-id": os.getenv('KFJ_PIPELINE_ID', 'unknown')
            }
        },
        "spec": {
            "ports": [
                {
                    "name": "client",
                    "protocol": "TCP",
                    "port": 10001,
                    "targetPort": 10001
                },
                {
                    "name": "dashboard",
                    "protocol": "TCP",
                    "port": 8265,
                    "targetPort": 8265
                },
                {
                    "name": "redis",
                    "protocol": "TCP",
                    "port": 6379,
                    "targetPort": 6379
                }
            ],
            "selector": {
                "component": name
            }
        }
    }
    return service_json

# @pysnooper.snoop()
def create_header_deploy(name):
    header_deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels":{
                "run-id":os.getenv('KFJ_RUN_ID','unknown'),
                "run-rtx":os.getenv('KFJ_RUNNER','unknown'),
                "pipeline-rtx": os.getenv('KFJ_CREATOR', 'unknown'),
                "task-id":os.getenv('KFJ_TASK_ID','unknown'),
                "pipeline-id": os.getenv('KFJ_PIPELINE_ID', 'unknown')
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "component": name,
                    "type": "ray"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "pipeline-id": KFJ_PIPELINE_ID,
                        "pipeline-name": KFJ_PIPELINE_NAME,
                        "task-name": KFJ_TASK_NAME,
                        'rtx-user': KFJ_RUNNER,
                        "component": name,
                        "type": "ray",
                        "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),
                    }
                },
                "spec": {
                    "restartPolicy": "Always",
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
                                    "weight": 5,
                                    "podAffinityTerm": {
                                        "topologyKey": "kubernetes.io/hostname",
                                        "labelSelector": {
                                            "matchLabels": {
                                                "component": name,
                                                "type":"ray"
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "containers": [
                        {
                            "name": "ray-head",
                            "image": KFJ_TASK_IMAGES,
                            "imagePullPolicy": "Always",
                            "workingDir": workingDir,
                            "command": [
                                "/bin/bash",
                                "-c",
                                "%s ray start --head --port=6379 --num-cpus=$MY_CPU_REQUEST --block"%INIT_FILE
                            ],
                            "ports": [
                                {
                                    "containerPort": 6379
                                },
                                {
                                    "containerPort": 10001
                                },
                                {
                                    "containerPort": 8265
                                }
                            ],
                            "volumeMounts": k8s_volume_mounts,
                            "env": [
                                {
                                    "name": "MY_CPU_REQUEST",
                                    "valueFrom": {
                                        "resourceFieldRef": {
                                            "resource": "requests.cpu"
                                        }
                                    }
                                }
                            ],
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
    }

    return header_deploy


def create_worker_deploy(header_name,worker_name):
    worker_deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": worker_name,
            "labels": {
                "run-id":os.getenv('KFJ_RUN_ID','unknown'),
                "run-rtx":os.getenv('KFJ_RUNNER','unknown'),
                "pipeline-rtx": os.getenv('KFJ_CREATOR', 'unknown'),
                "task-id":os.getenv('KFJ_TASK_ID','unknown'),
                "pipeline-id": os.getenv('KFJ_PIPELINE_ID', 'unknown')
            }
        },
        "spec": {
            "replicas": NUM_WORKER,
            "selector": {
                "matchLabels": {
                    "component": worker_name,
                    "type": "ray"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "pipeline-id": KFJ_PIPELINE_ID,
                        "pipeline-name": KFJ_PIPELINE_NAME,
                        "task-name": KFJ_TASK_NAME,
                        'rtx-user': KFJ_RUNNER,
                        "component": worker_name,
                        "type": "ray",
                        "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),

                    }
                },

                "spec": {
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
                                                "component": worker_name
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "imagePullSecrets": [
                        {
                            "name": "hubsecret"
                        }
                    ],
                    "restartPolicy": "Always",
                    "volumes": k8s_volumes,
                    "containers": [
                        {
                            "name": "ray-worker",
                            "image": KFJ_TASK_IMAGES,
                            "imagePullPolicy": "Always",
                            "workingDir": workingDir,
                            "command": [
                                "/bin/bash",
                                "-c",
                                "%s ray start --num-cpus=$MY_CPU_REQUEST --address=$RAY_HEAD_SERVICE_HOST:6379 --block"%INIT_FILE
                            ],
                            "volumeMounts": k8s_volume_mounts,
                            "env": [
                                {
                                    "name": "MY_CPU_REQUEST",
                                    "valueFrom": {
                                        "resourceFieldRef": {
                                            "resource": "requests.cpu"
                                        }
                                    }
                                },
                                {
                                    "name": "RAY_HEAD_SERVICE_HOST",
                                    "value": header_name
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": KFJ_TASK_RESOURCE_CPU,
                                    "memory": KFJ_TASK_RESOURCE_MEMORY
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
    }

    if int(gpu_num):
        worker_deploy['spec']['template']['spec']['containers'][0]['resources']['requests'][GPU_RESOURCE_NAME] = int(gpu_num)
        worker_deploy['spec']['template']['spec']['containers'][0]['resources']['limits'][GPU_RESOURCE_NAME] = int(gpu_num)
    else:
        # 添加禁用指令
        worker_deploy['spec']['template']['spec']['containers'][0]['env'].append({
            "name":"NVIDIA_VISIBLE_DEVICES",
            "value":"none"
        })

    return worker_deploy


# @pysnooper.snoop()
def wait_for_nodes():
    # Wait for all nodes to join the cluster.
    while True:
        resources = ray.cluster_resources()
        node_keys = [key for key in resources if "node" in key]
        num_nodes = sum(resources[node_key] for node_key in node_keys)
        if num_nodes < NUM_WORKER:
            print("{} nodes have joined so far, waiting for {} more.".format(num_nodes, NUM_WORKER - num_nodes))
            sys.stdout.flush()
            time.sleep(1)
        else:
            break


# @pysnooper.snoop()
def launcher_cluster(deal=None):
    # 清理一下之前存在的
    try:
        print('begin delete old header service')
        k8s_client.v1.delete_namespaced_service(HEADER_NAME, KFJ_NAMESPACE)
    except Exception as e1:
        pass
        print(e1)

    try:
        print('begin delete old header deployment')
        k8s_client.AppsV1Api.delete_namespaced_deployment(HEADER_NAME, KFJ_NAMESPACE)
    except Exception as e1:
        pass
        print(e1)

    try:
        print('begin delete old worker deployment')
        k8s_client.AppsV1Api.delete_namespaced_deployment(WORKER_NAME, KFJ_NAMESPACE)
    except Exception as e1:
        pass
        print(e1)
    time.sleep(3)

    if deal=='create':
        header_service = create_header_service(HEADER_NAME)
        header_deploy = create_header_deploy(HEADER_NAME)
        worker_deploy = create_worker_deploy(HEADER_NAME,WORKER_NAME)
        try:
            print(KFJ_NAMESPACE)
            print(header_service)
            print('begin create ray header service,%s ' % datetime.datetime.now())
            k8s_client.v1.create_namespaced_service(KFJ_NAMESPACE, header_service, pretty='true')
            print('begin create ray header deployment,%s ' % datetime.datetime.now())
            print(header_deploy)
            k8s_client.AppsV1Api.create_namespaced_deployment(KFJ_NAMESPACE, header_deploy, pretty='true')
            print('begin create ray worker deployment,%s ' % datetime.datetime.now())
            print(worker_deploy)
            k8s_client.AppsV1Api.create_namespaced_deployment(KFJ_NAMESPACE, worker_deploy, pretty='true')
            # 等待创建完成
            time.sleep(20)
            header_host = "%s:10001" % HEADER_NAME
            print('begin connect ray cluster %s,%s ' % (header_host,datetime.datetime.now()))

            ray.util.connect(header_host,connection_retries=20)
            wait_for_nodes()
            print('ray cluster all node ready,%s ' % datetime.datetime.now())

        except Exception as e:
            print(e)
            try:
                print('begin delete error header service')
                k8s_client.v1.delete_namespaced_service(HEADER_NAME, KFJ_NAMESPACE)
            except Exception as e1:
                pass
                # print(e1)
            try:
                print('begin delete error header deployment')
                k8s_client.AppsV1Api.delete_namespaced_deployment(HEADER_NAME, KFJ_NAMESPACE)
            except Exception as e1:
                pass
                # print(e1)
            try:
                print('begin delete error worker deployment')
                k8s_client.AppsV1Api.delete_namespaced_deployment(WORKER_NAME, KFJ_NAMESPACE)
            except Exception as e1:
                pass
                print(e1)
            # 如果出现错误，报错退出。不进行下一步代码
            raise e


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="build component")
    arg_parser.add_argument('--num_workers', type=int, required=False, help="workers的数量", default=3)
    arg_parser.add_argument('--workdir', type=str, required=False, help="启动目录", default='/')
    arg_parser.add_argument('--deal', type=str, required=False, help="创建集群还是删除集群", default='create')
    arg_parser.add_argument('--init', type=str, required=False, help="每个worker的初始化脚本，用来安装环境", default='')
    args = arg_parser.parse_args()
    print('NUM_WORKER',args.num_workers)
    print('INIT_FILE',args.init)

    if args.init.strip() and not os.path.exists(args.init):
        print('init file not exist')
        exit(1)
    workingDir = args.workdir
    NUM_WORKER = int(args.num_workers)
    if args.init.strip():
        INIT_FILE = "bash "+args.init.strip()+" && "
    launcher_cluster(deal=args.deal)
