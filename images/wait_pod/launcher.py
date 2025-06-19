import os
import time

import kubernetes
from kubernetes import config
from kubernetes import client
WAIT_SERVICE = os.getenv('WAIT_SERVICES','')
WAIT_PODS = os.getenv('WAIT_PODS','')
WAIT_LABELS = os.getenv('WAIT_LABELS','')
NAMESPACE = os.getenv('NAMESPACE','')

# 等待具有指定label的pod完成才启动
def wait_label(WAIT_LABELS):
    pass

# 等待指定service下的pod全部完成才启动
def wait_service(WAIT_SERVICE):
    pass


def wait_pods(WAIT_PODS):
    pod_names = WAIT_PODS.split(',')
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    all_status=[]
    while not all_status or not all(x == 'Running' for x in all_status):
        all_status=[]
        for pod_name in pod_names:
            status=''
            try:
                pod = v1.read_namespaced_pod(namespace=NAMESPACE,name=pod_name)
                status = pod.status.phase if pod and hasattr(pod, 'status') and hasattr(pod.status, 'phase') else ''
                print(pod_name,status)
            except Exception as e:
                print(e)
            all_status.append(status)
            time.sleep(5)

if WAIT_SERVICE:
    wait_service(WAIT_SERVICE)

if WAIT_PODS:
    wait_pods(WAIT_PODS)

if WAIT_LABELS:
    wait_pods(WAIT_LABELS)

# alpine:3.10 任何镜像都可以实现等待命令，不需要独立开发
# err=1;for i in $(seq 100); do if nslookup
#           pytorchjob-test-b348-master-0; then err=0 && break; fi;echo waiting
#           for master; sleep 2; done; exit $err