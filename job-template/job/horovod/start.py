import time,datetime,logging,os,sys

dir_common = os.path.split(os.path.realpath(__file__))[0] + '/../'
sys.path.append(dir_common)   # 将根目录添加到系统目录,才能正常引用common文件夹

import argparse
import requests
import os,sys,datetime,time,json

from job.pkgs.k8s.py_k8s import K8s
k8s_client = K8s()

from pathlib import Path

from threading import Thread
import pysnooper
import re
from kubernetes import client,config,watch
import uuid

base_dir = os.path.split(os.path.realpath(__file__))[0]
KFJ_NAMESPACE = os.getenv('KFJ_NAMESPACE', '')
KFJ_TASK_ID = os.getenv('KFJ_TASK_ID', '')
# KFJ_TASK_NAME = os.getenv('KFJ_TASK_NAME', '')
KFJ_TASK_NAME = "mpijob-" + str(uuid.uuid1())
task_node_selectors = re.split(',|;|\n|\t', os.getenv('KFJ_TASK_NODE_SELECTOR', ''))
KFJ_TASK_NODE_SELECTOR = {}
for task_node_selector in task_node_selectors:
    KFJ_TASK_NODE_SELECTOR[task_node_selector.split('=')[0]] = task_node_selector.split('=')[1]


KFJ_PIPELINE_ID = os.getenv('KFJ_PIPELINE_ID', '')
KFJ_RUN_ID = os.getenv('KFJ_RUN_ID', '')
KFJ_CREATOR = os.getenv('KFJ_CREATOR', '')
KFJ_RUNNER = os.getenv('KFJ_RUNNER')
KFJ_PIPELINE_NAME = os.getenv('KFJ_PIPELINE_NAME', '')
KFJ_TASK_IMAGES = os.getenv('KFJ_TASK_IMAGES', '')
KFJ_TASK_VOLUME_MOUNT = os.getenv('KFJ_TASK_VOLUME_MOUNT', '')
KFJ_TASK_RESOURCE_CPU = os.getenv('KFJ_TASK_RESOURCE_CPU', '')
KFJ_TASK_RESOURCE_MEMORY = os.getenv('KFJ_TASK_RESOURCE_MEMORY', '')
NUM_WORKER = 3
COMMAND=''
WORK_IMAGES='csighub.tencentyun.com/tme-kubeflow/horovod:cpu-20210401'


k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)


# k8s_volumes.append(
#     {
#         "name": "dshm",
#         "emptyDir": {
#             "medium": "Memory"
#         }
#     }
# )

print(k8s_volumes)
print(k8s_volume_mounts)

GPU_TYPE= os.getenv('KFJ_GPU_TYPE', 'NVIDIA')
GPU_RESOURCE= os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
print(GPU_TYPE,GPU_RESOURCE)

CRD_INFO={
    "group": "kubeflow.org",
    "version": "v1",
    "plural": "mpijobs",
    'kind': 'MPIJob',
    "timeout": 60 * 60 * 24 * 2
}



# @pysnooper.snoop()
def make_mpijob(name):
    mpijob={
        "apiVersion": "kubeflow.org/v1",
        "kind": "MPIJob",
        "metadata": {
            "name": name,
            "namespace":KFJ_NAMESPACE,
            "labels": {
                "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),
                "run-rtx": os.getenv('KFJ_RUNNER', 'unknown'),
                "pipeline-rtx": os.getenv('KFJ_CREATOR', 'unknown'),
                "task-id": os.getenv('KFJ_TASK_ID', 'unknown'),
                "pipeline-id": os.getenv('KFJ_PIPELINE_ID', 'unknown')
            }
        },
        "spec": {
            "slotsPerWorker": 1,
            "cleanPodPolicy": "Running",
            "mpiReplicaSpecs": {
                "Launcher": {
                    "replicas": 1,
                    "template": {
                        "metadata": {
                            "labels": {
                                "pipeline-id": KFJ_PIPELINE_ID,
                                "pipeline-name": KFJ_PIPELINE_NAME,
                                "task-name": KFJ_TASK_NAME,
                                'rtx-user': KFJ_RUNNER,
                                "component": name,
                                "type": "mpijob",
                                "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),
                            }
                        },
                        "spec": {
                            "volumes": k8s_volumes,
                            "imagePullSecrets": [
                                {
                                    "name": "hubsecret"
                                }
                            ],

                            "containers": [
                                {
                                    "image": WORK_IMAGES,
                                    "name": "mpi-launcher",
                                    "command": [
                                        "mpirun"
                                    ],
                                    "args": [
                                        "-np",
                                        str(NUM_WORKER),
                                        "--allow-run-as-root",
                                        "-bind-to",
                                        "none",
                                        "-map-by",
                                        "slot",
                                        "-x",
                                        "LD_LIBRARY_PATH",
                                        "-x",
                                        "PATH",
                                        "-mca",
                                        "pml",
                                        "ob1",
                                        "-mca",
                                        "btl",
                                        "^openib"

                                    ]+[item.strip() for item in COMMAND.split(' ') if item.strip()],
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
                                    },
                                    "volumeMounts": k8s_volume_mounts,
                                }
                            ]
                        }
                    }
                },
                "Worker": {
                    "replicas": NUM_WORKER,
                    "template": {
                        "metadata": {
                            "labels": {
                                "pipeline-id": KFJ_PIPELINE_ID,
                                "pipeline-name": KFJ_PIPELINE_NAME,
                                "task-name": KFJ_TASK_NAME,
                                'rtx-user': KFJ_RUNNER,
                                "component": name,
                                "type": "mpijob",
                                "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),
                            }
                        },
                        "spec": {
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
                                                        "type":"mpijob"
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                            "containers": [
                                {
                                    "image": WORK_IMAGES,
                                    "name": "mpi-worker",
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
                                    },
                                    "volumeMounts": k8s_volume_mounts,
                                }
                            ]
                        }
                    }
                }
            }
        }
    }


    #
    # if GPU_RESOURCE:
    #     mpijob['spec']['mpiReplicaSpecs']['Worker']['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]
    #     mpijob['spec']['mpiReplicaSpecs']['Worker']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = GPU_RESOURCE.split(',')[0]

    return mpijob



# 实时跟踪指定pod日志，直到pod结束
def watch_pod_log(name,namespace,container='main'):
    print('begin follow log')
    w = watch.Watch()
    for event in w.stream(client.CoreV1Api().read_namespaced_pod_log, name=name, namespace=namespace,container=container):
        print(event)

    print('end follow log')


# @pysnooper.snoop(watch_explode=('mpijob_json',))
def main():
    k8s_client = K8s()

    # 删除旧的mpi
    if KFJ_RUN_ID:
        print('begin delete old mpijob: run-id %s'%KFJ_RUN_ID)
        k8s_client.delete_crd(group=CRD_INFO['group'],
                              version=CRD_INFO['version'],
                              plural=CRD_INFO['plural'],
                              namespace=KFJ_NAMESPACE,
                              labels={"run-id":KFJ_RUN_ID})
        time.sleep(60)

    mpijob_json = make_mpijob(KFJ_TASK_NAME)
    print('begin create new mpijob: run-id %s' % KFJ_TASK_NAME)
    k8s_client.create_crd(
        group=CRD_INFO['group'],
        version=CRD_INFO['version'],
        plural=CRD_INFO['plural'],
        namespace=KFJ_NAMESPACE,
        body=mpijob_json
    )
    # 等待创建完成
    time.sleep(200)

    pods = k8s_client.get_pods(namespace=KFJ_NAMESPACE,labels={
        "mpi_job_name": KFJ_TASK_NAME
    })
    if pods:
        pod=pods[0]
        print('begin listen mpijob launcher pod %s' % pod['name'])
        k8s_client.watch_pod_log(name=pod['name'],namespace=KFJ_NAMESPACE)
        crd = k8s_client.get_one_crd(
            group=CRD_INFO['group'],
            version=CRD_INFO['version'],
            plural=CRD_INFO['plural'],
            namespace=KFJ_NAMESPACE,
            name=KFJ_TASK_NAME
        )
        print('begin delete mpijob %s' % KFJ_TASK_NAME)
        # 删除旧的mpi
        if KFJ_RUN_ID:
            k8s_client.delete_crd(group=CRD_INFO['group'],
                                  version=CRD_INFO['version'],
                                  plural=CRD_INFO['plural'],
                                  namespace=KFJ_NAMESPACE,
                                  labels={"run-id": KFJ_RUN_ID})
        print(crd)
        if crd['status']=='Succeeded':
            exit(0)
        else:
            exit(1)
    else:
        print('cluster fail build')
        print('begin delete mpijob %s' % KFJ_TASK_NAME)
        # 删除旧的mpi
        if KFJ_RUN_ID:
            k8s_client.delete_crd(group=CRD_INFO['group'],
                                  version=CRD_INFO['version'],
                                  plural=CRD_INFO['plural'],
                                  namespace=KFJ_NAMESPACE,
                                  labels={"run-id": KFJ_RUN_ID})

        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mpi config')
    parser.add_argument('--num_worker', type=int, default=2, help='并行worker的数目 (default: 2)')
    parser.add_argument('--command', type=str, default='python /horovod/examples/tensorflow2/tensorflow2_mnist.py', help='启动命令')
    parser.add_argument('--work_images', type=str, default='ccr.ccs.tencentyun.com/cube-studio/horovod:20210401', help='worker镜像')

    args = parser.parse_args()
    print(args)
    NUM_WORKER = args.num_worker
    COMMAND = args.command
    WORK_IMAGES = args.work_images

    main()








