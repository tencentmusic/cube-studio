
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
KFJ_TASK_TIMEOUT = int(os.getenv('KFJ_TASK_TIMEOUT', 60 * 60 * 24 * 2))
KFJ_TASK_RESOURCE_CPU = os.getenv('KFJ_TASK_RESOURCE_CPU', '')
KFJ_TASK_RESOURCE_MEMORY = os.getenv('KFJ_TASK_RESOURCE_MEMORY', '')

INIT_FILE=''
crd_info={
    "group": "sparkoperator.k8s.io",
    "version": "v1beta2",
    'kind': 'SparkApplication',
    "plural": "sparkapplications",
    "timeout": 60 * 60 * 24 * 2
}

k8s_volumes, k8s_volume_mounts = k8s_client.get_volume_mounts(KFJ_TASK_VOLUME_MOUNT,KFJ_CREATOR)

print(k8s_volumes)
print(k8s_volume_mounts)

GPU_TYPE= os.getenv('KFJ_GPU_TYPE', 'NVIDIA')
GPU_RESOURCE= os.getenv('KFJ_TASK_RESOURCE_GPU', '0')
print(GPU_TYPE,GPU_RESOURCE)



def default_job_name():
    name = "sparkjob-" + KFJ_PIPELINE_NAME.replace('_','-')+"-"+uuid.uuid4().hex[:4]
    return name[0:54]


# @pysnooper.snoop()
def make_sparkjob(name,**kwargs):


    label={
        "run-id":KFJ_RUN_ID,
        "run-rtx":KFJ_RUNNER,
        "pipeline-rtx": KFJ_CREATOR,
        "pipeline-id": KFJ_PIPELINE_ID,
        "pipeline-name": KFJ_PIPELINE_NAME,
        "task-id": KFJ_TASK_ID,
        "task-name": KFJ_TASK_NAME,
    }

    spark_deploy = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "namespace": KFJ_NAMESPACE,
            "name": name,
            "labels":label
        },
        "spec": {
            "type": kwargs['code_type'],    # Java Python R Scala
            "mode": "cluster",   # client  cluster   in-cluster-client
            "proxyUser":KFJ_CREATOR,
            "image": kwargs['image'],
            "imagePullPolicy": "Always",
            "mainClass": kwargs['code_class'],  # Java/Scala
            "mainApplicationFile": kwargs['code_file'],  # JAR, Python, or R file
            "arguments":kwargs['code_arguments'],
            "sparkConf":kwargs['sparkConf'],
            "hadoopConf":kwargs['hadoopConf'],
            "nodeSelector":KFJ_TASK_NODE_SELECTOR,
            "sparkVersion": "3.1.1",
            "pythonVersion":"3",
            "batchScheduler": "kube-batch",
            "restartPolicy": {
                "type": "Never"
            },
            "timeToLiveSeconds":KFJ_TASK_TIMEOUT,
            "volumes": k8s_volumes,
            "driver": {
                "cores": int(KFJ_TASK_RESOURCE_CPU),
                "coreLimit": str(KFJ_TASK_RESOURCE_CPU),
                "memory": KFJ_TASK_RESOURCE_MEMORY,
                # "memoryLimit": KFJ_TASK_RESOURCE_MEMORY,
                "labels": label,
                "serviceAccount": "spark",
                "volumeMounts": k8s_volume_mounts
            },
            "executor": {
                "instances": int(kwargs['num_worker']),
                "cores": int(KFJ_TASK_RESOURCE_CPU),
                "coreLimit": str(KFJ_TASK_RESOURCE_CPU),
                "memory": KFJ_TASK_RESOURCE_MEMORY,
                # "memoryLimit": KFJ_TASK_RESOURCE_MEMORY,
                "labels": label,
                "volumeMounts": k8s_volume_mounts,
                "affinity":{
                    "podAntiAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "weight": 5,
                                "podAffinityTerm": {
                                    "topologyKey": "kubernetes.io/hostname",
                                    "labelSelector": {
                                        "matchLabels": {
                                            "task-name": KFJ_TASK_NAME,
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }


        }
    }
    print(spark_deploy)
    return spark_deploy


# @pysnooper.snoop()
def launch_sparkjob(name, **kwargs):

    if KFJ_RUN_ID:
        print('delete old spark, run-id %s'%KFJ_RUN_ID, flush=True)
        k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,labels={"run-id":KFJ_RUN_ID})
        time.sleep(10)
    # 删除旧的spark
    k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=KFJ_NAMESPACE, name=name)
    time.sleep(10)
    # 创建新的spark
    sparkjob_json = make_sparkjob(name=name,**kwargs)
    print('create new spark %s' % name, flush=True)
    k8s_client.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,body=sparkjob_json)
    time.sleep(10)
    while True:
        time.sleep(10)
        sparkjob = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=KFJ_NAMESPACE, name=name)

        if sparkjob:
            status = json.loads(sparkjob['status_more']).get('applicationState', {}).get("state", '').upper()
            if status=='COMPLETED' or 'FAILED' in status:
                break

    sparkjob = k8s_client.get_one_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=KFJ_NAMESPACE,name=name)
    print("sparkjob %s finished, status %s"%(name, sparkjob['status_more']))
    status = json.loads(sparkjob['status_more']).get('applicationState', {}).get("state", '').upper()
    if 'FAILED' in status:
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("sparkjob launcher")
    arg_parser.add_argument('--image', type=str, help="运行job的镜像", default='ccr.ccs.tencentyun.com/cube-studio/spark-operator:spark-v3.1.1')
    arg_parser.add_argument('--num_worker', type=int, help="运行job所在的机器", default=3)
    arg_parser.add_argument('--code_type', type=str, help="代码类型", default='')  # Java Python R Scala
    arg_parser.add_argument('--code_class', type=str, help="代码类型", default='')  #
    arg_parser.add_argument('--code_file', type=str, help="代码地址", default='')  #  local://,http://,hdfs://,s3a://,gcs://
    arg_parser.add_argument('--code_arguments', type=str, help="代码参数",default='')  #
    arg_parser.add_argument('--sparkConf', type=str, help="spark配置", default='')  #
    arg_parser.add_argument('--hadoopConf', type=str, help="hadoop配置", default='')  #
    # arg_parser.add_argument('--driver_memory', type=str, help="driver端的内存", default='2g')
    # arg_parser.add_argument('--executor_memory', type=str, help="executor端的内存", default='2g')
    # arg_parser.add_argument('--driver_cpu', type=str, help="driver端的cpu", default='2')
    # arg_parser.add_argument('--executor_cpu', type=str, help="executor端的cpu", default='2')

    args = arg_parser.parse_args().__dict__

    print("{} args: {}".format(__file__, args))

    sparkConf = [[x.split('=')[0], x.split('=')[1]] for x in args['sparkConf'].split('\n') if '=' in x]
    args['sparkConf'] = dict(zip([x[0] for x in sparkConf],[x[1] for x in sparkConf]))
    # args['sparkConf']['spark.driver.bindAddress']='0.0.0.0'  # k8s模式下不能用

    hadoopConf = [[x.split('=')[0], x.split('=')[1]] for x in args['hadoopConf'].split('\n') if '=' in x]
    args['hadoopConf'] = dict(zip([x[0] for x in hadoopConf], [x[1] for x in hadoopConf]))
    args['code_arguments'] = [x.strip() for x in args['code_arguments'].split(' ') if x.strip()]


    launch_sparkjob(name=default_job_name(),**args)


