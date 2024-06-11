import time, datetime, os
import re
from kubernetes import client
from kubernetes.client.models import v1_pod, v1_object_meta, v1_pod_spec, v1_deployment, v1_deployment_spec
import yaml
import json
import multiprocessing
import base64
from kubernetes import config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
import pysnooper
import traceback
import threading
import logging
from myapp import conf
from myapp.utils import core

class K8s():

    def __init__(self, file_path=None):  # kubeconfig
        if not file_path:
            file_path = conf.get('CLUSTERS',{}).get(conf.get('ENVIRONMENT'),{}).get('KUBECONFIG','')
        kubeconfig = os.getenv('KUBECONFIG', '')
        if file_path and os.path.exists(file_path) and ''.join(open(file_path).readlines()).strip():
            config.kube_config.load_kube_config(config_file=file_path)
        elif kubeconfig:
            config.kube_config.load_kube_config(config_file=kubeconfig)
        else:
            config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.v1beta1 = client.ExtensionsV1beta1Api()
        self.AppsV1Api = client.AppsV1Api()
        self.NetworkingV1Api = client.NetworkingV1Api()
        self.CustomObjectsApi = client.CustomObjectsApi()
        self.v1.api_client.configuration.verify_ssl = False  # 只能设置 /usr/local/lib/python3.9/dist-packages/kubernetes/client/configuration.py:   self.verify_ssl= True ---> False
        self.gpu_resource=conf.get('GPU_RESOURCE',{})
        self.vgpu_resource = conf.get('VGPU_RESOURCE', {})
        self.vgpu_drive_type = conf.get("VGPU_DRIVE_TYPE", "mgpu")

        self.get_gpu = core.get_gpu

    # 获取指定范围的pod
    # @pysnooper.snoop()
    def get_running_pods(self, namespace=None):
        all_pods=[]
        all_endpoints = self.v1.list_namespaced_endpoints(namespace=namespace)  # 先查询入口点，
        subsets = all_endpoints.subsets
        addresses = subsets[0].addresses  # 只取第一个子网
        for address in addresses:
            pod_name_temp = address.target_ref.name
            pod = self.v1.read_namespaced_pod(name=pod_name_temp, namespace=namespace)
            all_pods.append(pod)

    # @pysnooper.snoop()
    def get_pods(self, namespace=None, service_name=None, pod_name=None, labels={},status=None):
        # print(namespace)
        back_pods = []
        try:
            all_pods = []
            # 如果只有命名空间
            if (namespace and not service_name and not pod_name and not labels):
                all_pods = self.v1.list_namespaced_pod(namespace).items
            # 如果有命名空间和pod名，就直接查询pod
            elif (namespace and pod_name):
                pod = self.v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                all_pods.append(pod)
            # 如果只有命名空间和服务名，就查服务下绑定的pod
            elif (namespace and service_name):  # 如果有命名空间和服务名
                all_endpoints = self.v1.read_namespaced_endpoints(service_name, namespace)  # 先查询入口点，
                subsets = all_endpoints.subsets
                addresses = subsets[0].addresses  # 只取第一个子网
                for address in addresses:
                    pod_name_temp = address.target_ref.name
                    pod = self.v1.read_namespaced_pod(name=pod_name_temp, namespace=namespace)
                    all_pods.append(pod)
            elif (namespace and status):
                if status.lower()=='running':
                    all_endpoints = self.v1.list_namespaced_endpoints(namespace=namespace)  # 先查询入口点，
                    subsets = all_endpoints.subsets
                    addresses = subsets[0].addresses  # 只取第一个子网
                    for address in addresses:
                        pod_name_temp = address.target_ref.name
                        pod = self.v1.read_namespaced_pod(name=pod_name_temp, namespace=namespace)
                        all_pods.append(pod)
                else:
                    src_pods = self.v1.list_namespaced_pod(namespace).items
                    for pod in src_pods:
                        if pod.status and pod.status.phase == status:
                            all_pods.append(pod)


            elif (namespace and labels):
                src_pods = self.v1.list_namespaced_pod(namespace).items
                for pod in src_pods:
                    pod_labels = pod.metadata.labels
                    is_des_pod = True
                    for key in labels:
                        if key not in pod_labels or pod_labels[key] != labels[key]:
                            is_des_pod = False
                            break
                    if is_des_pod:
                        all_pods.append(pod)

            for pod in all_pods:
                # print(pod)
                metadata = pod.metadata
                status = pod.status.phase if pod and hasattr(pod, 'status') and hasattr(pod.status, 'phase') else ''
                # 如果是running 也分为重启运行中
                if status.lower()=='running':
                    status = 'Running' if [x.status for x in pod.status.conditions if x.type == 'Ready' and x.status == 'True'] else 'CrashLoopBackOff'

                containers = pod.spec.containers
                # mem = [container.resources.requests for container in containers]
                memory = [self.to_memory_GB(container.resources.requests.get('memory','0G')) for container in containers if container.resources and container.resources.requests]
                cpu = [self.to_cpu(container.resources.requests.get('cpu', '0')) for container in containers if container.resources  and container.resources.requests]

                # gpu = [int(container.resources.requests.get('nvidia.com/gpu', '0')) for container in containers if container.resources and container.resources.requests]
                vgpu = [float(container.resources.requests.get('tencent.com/vcuda-core', '0'))/100 for container in containers if container.resources and container.resources.requests]
                vgpu += [float(container.resources.requests.get('tke.cloud.tencent.com/qgpu-core', '0'))/100 for container in containers if container.resources and container.resources.requests]
                # 获取gpu异构资源占用
                ai_resource={}
                for name in self.gpu_resource:
                    resource = self.gpu_resource[name]
                    gpu = [int(container.resources.requests.get(resource, '0')) for container in containers if container.resources and container.resources.requests]
                    ai_resource[name]=sum(gpu)
                ai_resource['gpu']=ai_resource.get('gpu',0)+sum(vgpu)

                node_selector = {}
                try:
                    # aa=client.V1NodeSelector
                    match_expressions = pod.spec.affinity.node_affinity.required_during_scheduling_ignored_during_execution.node_selector_terms
                    match_expressions = [ex.match_expressions for ex in match_expressions]
                    match_expressions = match_expressions[0]
                    for match_expression in match_expressions:
                        if match_expression.operator == 'In':
                            node_selector[match_expression.key] = match_expression.values[0]
                        if match_expression.operator == 'Equal':
                            node_selector[match_expression.key] = match_expression.values

                except Exception:
                    pass
                    # print(e)
                if pod.spec.node_selector:
                    node_selector.update(pod.spec.node_selector)

                username = ''
                if pod.metadata.labels:
                    username = pod.metadata.labels.get('run-rtx', '')
                    if not username:
                        username = pod.metadata.labels.get('user', '')
                    if not username:
                        username = pod.metadata.labels.get('rtx-user', '')

                temp = {
                    'name': metadata.name,
                    "username": username,
                    'host_ip': pod.status.host_ip,
                    'pod_ip': pod.status.pod_ip,
                    'status': status,  # 每个容器都正常才算正常
                    'status_more': pod.status.to_dict(),  # 无法json序列化
                    'node_name': pod.spec.node_name,
                    "labels": metadata.labels if metadata.labels else {},
                    "annotations": metadata.annotations if metadata.annotations else {},
                    "memory": sum(memory),
                    "cpu": sum(cpu),
                    # "gpu": sum(gpu) + sum(vgpu),
                    "start_time": (metadata.creation_timestamp + datetime.timedelta(hours=8)).replace(tzinfo=None),   # 时间格式
                    "node_selector": node_selector
                }
                temp.update(ai_resource)



                back_pods.append(temp)
            # print(back_pods)
            return back_pods

        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
            return back_pods

    def get_pod_event(self, namespace, pod_name):
        events = [item.to_dict() for item in self.v1.list_namespaced_event(namespace, field_selector=f'involvedObject.name={pod_name}').items]
        for event in events:
            event['time'] = (event['first_timestamp'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('first_timestamp', None) else None
            if not event['time']:
                event['time'] = (event['event_time'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('event_time', None) else None
        return events

    # 获取 指定服务，指定命名空间的下面的endpoint
    def get_pod_humanized(self, namespace, pod_name):
        try:
            pod = self.v1.read_namespaced_pod(namespace=namespace, name=pod_name)
            if pod:
                from kubernetes.client import ApiClient
                pod = ApiClient().sanitize_for_serialization(pod)
                if 'managedFields' in pod.get('metadata', {}):
                    del pod['metadata']['managedFields']
                if 'ownerReferences' in pod.get('metadata', {}):
                    del pod['metadata']['ownerReferences']
                # print(json.dumps(pod,indent=4,ensure_ascii=False))
                return pod
        except Exception as e:
            print(e)

    # 获取 指定服务，指定命名空间的下面的endpoint
    def get_pod_ip(self, namespace, service_name):
        try:
            all_pods = self.get_pods(namespace=namespace, service_name=service_name)
            all_pod_ip = []
            if (all_pods):
                for pod in all_pods:
                    all_pod_ip.append(pod['pod_ip'])
                # print(all_pod_ip)
            return all_pod_ip
        except Exception as e:
            print(e)
            return None

    # 指定命名空间，指定服务名，指定pod名称，指定状态，删除重启pod。status为运行状态,True  或者False
    def delete_pods(self, namespace=None, service_name=None, pod_name=None, status=None, labels=None):
        if not namespace:
            return []
        all_pods = self.get_pods(namespace=namespace, pod_name=pod_name, service_name=service_name, labels=labels)
        if status:
            all_pods = [pod for pod in all_pods if pod['status'] == status]
        try:
            for pod in all_pods:
                self.v1.delete_namespaced_pod(pod['name'], namespace, grace_period_seconds=0)
                print('delete pod %s' % pod['name'])
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
        return all_pods

    # @pysnooper.snoop()
    def get_all_node_allocated_resources(self):
        nodes_resource = {}
        try:
            pods = self.v1.list_pod_for_all_namespaces(watch=False).items

            for pod in pods:
                if not pod.status or pod.status.phase != 'Running':
                    continue
                containers = pod.spec.containers
                memory = [self.to_memory_GB(container.resources.requests.get('memory', '0G')) for container in containers if container.resources and container.resources.requests]
                cpu = [self.to_cpu(container.resources.requests.get('cpu', '0')) for container in containers if container.resources and container.resources.requests]
                # gpu = [int(container.resources.requests.get('nvidia.com/gpu', '0')) for container in containers if container.resources and container.resources.requests]
                vgpu = [float(container.resources.requests.get('tencent.com/vcuda-core', '0')) / 100 for container in containers if container.resources and container.resources.requests]
                vgpu += [float(container.resources.requests.get('tke.cloud.tencent.com/qgpu-core', '0')) / 100 for container in containers if container.resources and container.resources.requests]
                node_name = pod.spec.node_name
                if node_name not in nodes_resource:
                    nodes_resource[node_name] = {
                        "used_memory": 0,
                        "used_cpu": 0,
                        "used_gpu": 0
                    }
                nodes_resource[node_name]['used_memory'] += sum(memory)
                nodes_resource[node_name]['used_cpu'] += sum(cpu)

                # 获取gpu异构资源占用
                for name in self.gpu_resource:
                    resource = self.gpu_resource[name]
                    gpu = [int(container.resources.requests.get(resource, '0')) for container in containers if container.resources and container.resources.requests]
                    nodes_resource[node_name]["used_"+name] = nodes_resource[node_name].get("used_"+name,0)+sum(gpu)
                    # print(pod.metadata.name,"used_"+name,sum(gpu))
                nodes_resource[node_name]['used_gpu'] = nodes_resource[node_name].get('used_gpu', 0) + sum(vgpu)

            for node_name in nodes_resource:
                node_resource = nodes_resource[node_name]
                # print(node_resource)
                node_resource['used_memory'] = int(node_resource['used_memory'])
                node_resource['used_cpu'] = int(node_resource['used_cpu'])
                node_resource['used_gpu'] = round(node_resource['used_gpu'], 1)

        except Exception as e:
            logging.error('Traceback: %s', traceback.format_exc())
            print(e)

        return nodes_resource

    def get_node_event(self, node_name):
        node = self.get_node(name=node_name)
        events = [item.to_dict() for item in self.v1.list_event_for_all_namespaces().items]   # field_selector=f'source.host={node["hostip"]}'
        for event in events:
            event['time'] = (event['first_timestamp'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('first_timestamp', None) else None
            if not event['time']:
                event['time'] = (event['event_time'] + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if event.get('event_time', None) else None
        back_event = []
        for event in events:
            if event.get("source", {}).get("host", '') == node["hostip"]:
                back_event.append(event)
        return back_event

    # 获取指定label的nodeip列表
    # @pysnooper.snoop()
    def get_node(self, label=None, name=None, ip=None):
        try:
            back_nodes = []
            all_node = self.v1.list_node(label_selector=label).items
            # print(all_node)
            for node in all_node:
                try:
                    back_node = {}
                    # 获取gpu异构资源占用
                    ai_resource = {}
                    for gpu_mfrs in self.gpu_resource:
                        resource = self.gpu_resource[gpu_mfrs]
                        ai_resource[gpu_mfrs] = int(node.status.allocatable.get(resource, '0'))

                    # print(node.status.conditions)
                    adresses = node.status.addresses
                    back_node['cpu'] = int(self.to_cpu(node.status.allocatable.get('cpu', '0')))
                    back_node['memory'] = int(self.to_memory_GB(node.status.allocatable.get('memory', '0')))
                    # back_node['gpu'] = int(node.status.allocatable.get('nvidia.com/gpu', '0'))
                    back_node['labels'] = node.metadata.labels
                    back_node['name'] = node.metadata.name
                    back_node['create_time'] = node.metadata.creation_timestamp
                    back_node['node_info'] = node.status.node_info.to_dict()
                    back_node['status'] = 'Ready' if [x.status for x in node.status.conditions if x.type=='Ready' and x.status=='True'] else 'Unknown'
                    # print(back_node['status'])
                    back_node.update(ai_resource)

                    for address in adresses:
                        if address.type == 'InternalIP':
                            back_node['hostip'] = address.address
                            break

                    if name and back_node['name'] == name:
                        back_nodes.append(back_node)
                    elif ip and back_node['hostip'] == ip:
                        back_nodes.append(back_node)
                    elif not name and not ip:
                        back_nodes.append(back_node)
                except Exception as e1:
                    print(e1)

            return back_nodes
        except Exception as e:
            print(e)
            return []

    # 获取指定label的nodeip列表
    def label_node(self, ips, labels):
        try:
            all_node_ip = []
            all_node = self.v1.list_node().items

            for node in all_node:
                # print(node)
                adresses = node.status.addresses
                Hostname = ''
                InternalIP = ''
                for address in adresses:
                    if address.type == 'Hostname':
                        Hostname = address.address
                    if address.type == 'InternalIP':
                        InternalIP = address.address

                if InternalIP in ips:
                    body = {
                        "metadata": {
                            "labels": labels
                        }
                    }
                    self.v1.patch_node(Hostname, body)

            return all_node_ip
        except Exception as e:
            print(e)
            return None

    # 根据各种crd自定义的status结构，判断最终评定的status
    # @pysnooper.snoop()
    def get_crd_status(self, crd_object, group, plural):
        status = ''
        # workflows 使用最后一个node的状态为真是状态
        if plural == 'workflows':
            status = crd_object.get('status', {}).get('phase', '')
            if 'status' in crd_object and 'nodes' in crd_object['status']:
                keys = list(crd_object['status']['nodes'].keys())
                status = crd_object['status']['nodes'][keys[-1]]['phase']
                if status != 'Pending':
                    status = crd_object['status']['phase']
        elif plural == 'notebooks':
            if 'status' in crd_object and 'conditions' in crd_object['status'] and len(crd_object['status']['conditions']) > 0:
                status = crd_object['status']['conditions'][0]['type']
        elif plural == 'inferenceservices':
            status = 'unready'
            if 'status' in crd_object and 'conditions' in crd_object['status'] and len(crd_object['status']['conditions']) > 0:
                for condition in crd_object['status']['conditions']:
                    if condition['type'] == 'Ready' and condition['status'] == 'True':
                        status = 'ready'
        elif plural == 'jobs' and group == 'batch.volcano.sh':
            status = 'unready'
            if 'status' in crd_object and 'state' in crd_object['status'] and 'phase' in crd_object['status']['state']:
                return crd_object['status']['state']['phase']
        else:
            if 'status' in crd_object and 'phase' in crd_object['status']:
                status = crd_object['status']['phase']
            elif 'status' in crd_object and 'conditions' in crd_object['status'] and len(
                    crd_object['status']['conditions']) > 0:
                status = crd_object['status']['conditions'][-1]['type']  # tfjob和experiment是这种结构
        return status

    # @pysnooper.snoop(watch_explode=('ya_str',))
    def get_one_crd_yaml(self, group, version, plural, namespace, name):
        try:
            crd_object = self.CustomObjectsApi.get_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, name=name)
            ya = yaml.load(json.dumps(crd_object))
            ya_str = yaml.safe_dump(ya, default_flow_style=False)
            return ya_str
        except Exception as e:
            print(e)
        return ''

    # @pysnooper.snoop(watch_explode=('crd_object'))
    def get_one_crd(self, group, version, plural, namespace, name):
        try:
            crd_object = self.CustomObjectsApi.get_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural,name=name)
            if not crd_object:
                return {}

            # print(crd_object['status']['conditions'][-1]['type'])
            status = self.get_crd_status(crd_object, group, plural)

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            back_object = {
                "name": crd_object['metadata']['name'],
                "namespace": crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations": json.dumps(crd_object['metadata']['annotations'], indent=4,
                                          ensure_ascii=False) if 'annotations' in crd_object['metadata'] else '{}',
                "labels": json.dumps(crd_object['metadata']['labels'], indent=4, ensure_ascii=False) if 'labels' in crd_object['metadata'] else '{}',
                "spec": json.dumps(crd_object['spec'], indent=4, ensure_ascii=False),
                "create_time": creat_time,
                "status": status,
                "status_more": json.dumps(crd_object['status'], indent=4,ensure_ascii=False) if 'status' in crd_object else '{}'
            }

            # return
            return back_object
        except Exception as e:
            print(e)
            return {}

    # @pysnooper.snoop(watch_explode=())
    def get_crd(self, group, version, plural, namespace, label_selector=None, return_dict=None):
        crd_objects=[]
        try:
            if label_selector:
                crd_objects = self.CustomObjectsApi.list_namespaced_custom_object(group=group,version=version,namespace=namespace,plural=plural,label_selector=label_selector)['items']
            else:
                crd_objects = self.CustomObjectsApi.list_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural)['items']
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
        back_objects=[]
        for crd_object in crd_objects:
            # print(crd_object['status']['conditions'][-1]['type'])
            status = self.get_crd_status(crd_object, group, plural)

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            finish_time=''
            if 'status' in crd_object and 'finishedAt' in crd_object['status'] and crd_object['status']['finishedAt']:
                finish_time = crd_object['status']['finishedAt'].replace('T', ' ').replace('Z', '')
                finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            elif 'status' in crd_object and 'completionTime' in crd_object['status'] and crd_object['status']['completionTime']:
                finish_time = crd_object['status']['completionTime'].replace('T', ' ').replace('Z', '')
                finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            # vcjob的结束时间
            elif 'status' in crd_object and 'state' in crd_object['status'] and 'lastTransitionTime' in crd_object['status']['state']:
                if crd_object['status']['state'].get('phase','')=='Completed' or crd_object['status']['state'].get('phase','')=='Aborted' or crd_object['status']['state'].get('phase','')=='Failed' or crd_object['status']['state'].get('phase','')=='Terminated':
                    finish_time = crd_object['status']['state']['lastTransitionTime'].replace('T', ' ').replace('Z', '')
                    finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')


            back_object={
                "name":crd_object['metadata']['name'],
                "namespace":crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations":json.dumps(crd_object['metadata']['annotations'],indent=4,ensure_ascii=False) if 'annotations' in crd_object['metadata'] else '',
                "labels": json.dumps(crd_object['metadata']['labels'], indent=4, ensure_ascii=False) if 'labels' in crd_object['metadata'] else '{}',
                "spec": json.dumps(crd_object['spec'], indent=4, ensure_ascii=False),
                "create_time": creat_time,
                "finish_time": finish_time,
                "status": status,
                "status_more": json.dumps(crd_object['status'], indent=4, ensure_ascii=False) if 'status' in crd_object else ''
            }
            back_objects.append(back_object)
            # return
        if return_dict != None:
            return_dict[namespace] = back_objects
        return back_objects

    # @pysnooper.snoop(watch_explode=())
    def get_crd_all_namespaces(self, group, version, plural, pool=False):
        all_namespace = self.v1.list_namespace().items
        all_namespace = [namespace.metadata.name for namespace in all_namespace]
        back_objects = []
        jobs = []
        if pool:
            from multiprocessing import Manager
            manager = Manager()
            return_dict = manager.dict()
            for namespace in all_namespace:
                p = multiprocessing.Process(target=self.get_crd, args=(group, version, plural, namespace, return_dict))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join(timeout=5)
            for namespace_name in return_dict:
                for crd_object in return_dict[namespace_name]:
                    back_objects.append(crd_object)
            return back_objects
        else:
            for namespace in all_namespace:
                crds = self.get_crd(group=group, version=version, plural=plural, namespace=namespace)
                for crd_object in crds:
                    back_objects.append(crd_object)
            return back_objects

    # @pysnooper.snoop(watch_explode=())
    def delete_crd(self, group, version, plural, namespace, name='', labels=None):
        if name:
            try:
                delete_body = client.V1DeleteOptions(grace_period_seconds=0)
                self.CustomObjectsApi.delete_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, name=name, body=delete_body)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                logging.error('Traceback: %s', traceback.format_exc())
                print(e)
            return [name]
        elif labels:
            back_name = []
            crds = self.get_crd(group=group, version=version, plural=plural, namespace=namespace)
            for crd in crds:
                if crd['labels']:
                    crd_labels = json.loads(crd['labels'])
                    for key in labels:
                        if key in crd_labels and labels[key] == crd_labels[key]:
                            try:
                                delete_body = client.V1DeleteOptions(grace_period_seconds=0)
                                self.CustomObjectsApi.delete_namespaced_custom_object(group=group, version=version,
                                                                                      namespace=namespace,
                                                                                      plural=plural, name=crd['name'],
                                                                                      body=delete_body)
                            except ApiException as api_e:
                                if api_e.status != 404:
                                    print(api_e)
                            except Exception as e:
                                print(e)
                                logging.error('Traceback: %s', traceback.format_exc())
                            back_name.append(crd['name'])
            return back_name

    # @pysnooper.snoop()
    def delete_workflow(self, all_crd_info, namespace, run_id):
        if not run_id:
            return None

        if run_id:

            # 删除workflow
            crd_info = all_crd_info['workflow']
            try:
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'],
                    plural=crd_info['plural'], namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除tfjob
            try:
                crd_info = all_crd_info['tfjob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除pytorchjob
            try:
                crd_info = all_crd_info['pytorchjob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除mpijob
            try:
                crd_info = all_crd_info['mpijob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除vcjob
            try:
                crd_info = all_crd_info['vcjob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除sparkjob
            try:
                crd_info = all_crd_info['sparkjob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除paddlejob
            try:
                crd_info = all_crd_info['paddlejob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除mxjob
            try:
                crd_info = all_crd_info['mxjob']
                self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除deployment
            try:
                self.delete_deployment(namespace=namespace, labels={'run-id': run_id})
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除stss
            try:
                stss = self.AppsV1Api.list_namespaced_stateful_set(namespace=namespace, label_selector="run-id=%s" % str(run_id)).items
                if stss:
                    for sts in stss:
                        self.AppsV1Api.delete_namespaced_stateful_set(namespace=namespace, name=sts.metadata.name, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除daemonsets
            try:
                daemonsets = self.AppsV1Api.list_namespaced_daemon_set(namespace=namespace, label_selector="run-id=%s" % str(run_id)).items
                if daemonsets:
                    for daemonset in daemonsets:
                        self.AppsV1Api.delete_namespaced_daemon_set(namespace=namespace, name=daemonset.metadata.name, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 删除service
            try:
                services = self.v1.list_namespaced_service(namespace=namespace, label_selector="run-id=%s" % str(run_id)).items
                if services:
                    for service in services:
                        self.v1.delete_namespaced_service(namespace=namespace, name=service.metadata.name, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
                logging.error('Traceback: %s', traceback.format_exc())

            # 不能删除pod，因为task的模板也是有这个run-id的，所以不能删除

    def delete_service(self, namespace, name=None, labels=None):
        if name:
            try:
                self.v1.delete_namespaced_service(name=name, namespace=namespace, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
        if labels:
            try:
                # 获取具有指定标签的服务
                services = self.v1.list_namespaced_service(namespace="your-namespace", label_selector=",".join([f"{k}={v}" for k, v in labels.items()])).items

                # 遍历每个服务
                for service in services:
                    # 删除服务
                    self.v1.delete_namespaced_service(name=service.metadata.name, namespace=service.metadata.namespace)
                    print(f"Deleted service: {service.metadata.name} in namespace: {service.metadata.namespace}")

            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
    #
    # @pysnooper.snoop()
    # def get_volume_mounts(self,volume_mount,username):
    #     k8s_volumes = []
    #     k8s_volume_mounts = []
    #     if volume_mount and ":" in volume_mount:
    #         volume_mount = volume_mount.strip()
    #         if volume_mount:
    #             volume_mounts_temp = re.split(',|;', volumdelete_workflowe_mount)
    #             volume_mounts_temp = [volume_mount_temp.strip() for volume_mount_temp in volume_mounts_temp if volume_mount_temp.strip()]
    #
    #             for volume_mount in volume_mounts_temp:
    #                 volume, mount = volume_mount.split(":")[0].strip(), volume_mount.split(":")[1].strip()
    #                 if "(pvc)" in volume:
    #                     pvc_name = volume.replace('(pvc)', '').replace(' ', '')
    #                     volumn_name = pvc_name.replace('_', '-').lower()
    #                     k8s_volumes.append(client.V1Volume(name=volumn_name,
    #                                                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
    #                                                            claim_name=pvc_name)))
    #                     k8s_volume_mounts.append(
    #                         client.V1VolumeMount(name=volumn_name, mount_path=os.path.join(mount, username),
    #                                              sub_path=username))
    #                 if "(hostpath)" in volume:
    #                     hostpath_name = volume.replace('(hostpath)', '').replace(' ', '')
    #                     temps = re.split('_|\.|/', hostpath_name)
    #                     temps = [temp for temp in temps if temp]
    #                     volumn_name = '-'.join(temps).lower()  # hostpath_name.replace('_', '-').replace('/', '-').replace('.', '-')
    #                     k8s_volumes.append(client.V1Volume(name=volumn_name,
    #                                                        host_path=client.V1HostPathVolumeSource(path=hostpath_name)))
    #                     k8s_volume_mounts.append(client.V1VolumeMount(name=volumn_name, mount_path=mount))
    #
    #                 if "(configmap)" in volume:
    #                     configmap_name = volume.replace('(configmap)', '').replace(' ', '')
    #                     volumn_name = configmap_name.replace('_', '-').replace('/', '-').replace('.', '-').lower()
    #                     k8s_volumes.append(client.V1Volume(name=volumn_name, host_path=client.V1ConfigMapVolumeSource(
    #                         name=configmap_name)))
    #                     k8s_volume_mounts.append(client.V1VolumeMount(name=volumn_name, mount_path=mount))
    #
    #     return k8s_volumes,k8s_volume_mounts

    # @pysnooper.snoop()
    @staticmethod
    def get_volume_mounts(volume_mount, username):
        k8s_volumes = []
        k8s_volume_mounts = []
        if volume_mount and ":" in volume_mount:
            volume_mount_new = volume_mount.strip()
            if volume_mount_new:
                volume_mounts_temp = re.split(',|;', volume_mount_new)
                volume_mounts_temp = [volume_mount_temp.strip() for volume_mount_temp in volume_mounts_temp if volume_mount_temp.strip()]

                for one_volume_mount in volume_mounts_temp:
                    volume, mount = one_volume_mount.split(":")[0].strip(), one_volume_mount.split(":")[1].strip()
                    if "(pvc)" in volume:
                        pvc_name = volume.replace('(pvc)', '').replace(' ', '')
                        volumn_name = pvc_name.replace('_', '-').lower()[-60:].strip('-')
                        k8s_volumes.append({
                            "name": volumn_name,
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        })
                        k8s_volume_mounts.append(
                            {
                                "name": volumn_name,
                                "mountPath": os.path.join(mount, username),
                                "subPath": username
                            }
                        )
                    if "(pvc-share)" in volume:
                        pvc_name = volume.replace('(pvc-share)', '').replace(' ', '')
                        volumn_name = pvc_name.replace('_', '-').lower()[-60:].strip('-')
                        k8s_volumes.append({
                            "name": volumn_name,
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        })
                        k8s_volume_mounts.append(
                            {
                                "name": volumn_name,
                                "mountPath": mount
                            }
                        )

                    # 外部挂载盘不挂载子目录
                    if "(storage)" in volume:
                        pvc_name = volume.replace('(storage)', '').replace(' ', '')
                        volumn_name = pvc_name.replace('_', '-').lower()[-60:].strip('-')
                        k8s_volumes.append({
                            "name": volumn_name,
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        })
                        k8s_volume_mounts.append(
                            {
                                "name": volumn_name,
                                "mountPath": mount,
                            }
                        )

                    if "(hostpath)" in volume:
                        hostpath_name = volume.replace('(hostpath)', '').replace(' ', '')
                        temps = re.split('_|\.|/', hostpath_name)
                        temps = [temp for temp in temps if temp]
                        volumn_name = '-'.join(temps).lower()[-60:].strip('-')  # hostpath_name.replace('_', '-').replace('/', '-').replace('.', '-')
                        k8s_volumes.append(
                            {
                                "name": volumn_name,
                                "hostPath": {
                                    "path": hostpath_name
                                }
                            }
                        )
                        k8s_volume_mounts.append({
                            "name": volumn_name,
                            "mountPath": mount
                        })

                    if "(configmap)" in volume:
                        configmap_name = volume.replace('(configmap)', '').replace(' ', '')
                        volumn_name = configmap_name.replace('_', '-').replace('/', '-').replace('.', '-').lower()[-60:].strip('-')
                        k8s_volumes.append({
                            "name": volumn_name,
                            "configMap": {
                                "name": configmap_name
                            }
                        })

                        k8s_volume_mounts.append({
                            "name": volumn_name,
                            "mountPath": mount
                        })
                    if "(secret)" in volume:
                        configmap_name = volume.replace('(secret)', '').replace(' ', '')
                        volumn_name = configmap_name.replace('_', '-').replace('/', '-').replace('.', '-').lower()[-60:].strip('-')
                        k8s_volumes.append({
                            "name": volumn_name,
                            "secret": {
                                "secretName": configmap_name
                            }
                        })

                        k8s_volume_mounts.append({
                            "name": volumn_name,
                            "mountPath": mount
                        })
                    if "(memory)" in volume:
                        memory_size = volume.replace('(memory)', '').replace(' ', '').lower().replace('g', '')
                        volumn_name = ('memory-%s' % memory_size)[-60:].strip('-')

                        k8s_volumes.append({
                            "name": volumn_name,
                            "emptyDir": {
                                "medium": "Memory",
                                "sizeLimit": "%sGi" % memory_size
                            }
                        })

                        k8s_volume_mounts.append({
                            "name": volumn_name,
                            "mountPath": mount
                        })

            if "/usr/share/zoneinfo/Asia/Shanghai" not in volume_mount:
                k8s_volumes.append(
                    {
                        "name": 'tz-config',
                        "hostPath": {
                            "path": '/usr/share/zoneinfo/Asia/Shanghai'
                        }
                    }
                )
                k8s_volume_mounts.append(
                    {
                        "name": 'tz-config',
                        "mountPath": '/etc/localtime'
                    }
                )
            if '/dev/shm' not in volume_mount:
                k8s_volume_mounts.append(
                    {
                        "name": 'dshm',
                        "mountPath": "/dev/shm"
                    }
                )
                k8s_volumes.append(
                    {
                        "name": "dshm",
                        "emptyDir": {
                            "medium": "Memory"
                        }
                    }
                )
        return k8s_volumes, k8s_volume_mounts


    # @pysnooper.snoop(watch_explode=())
    def make_container(self, name, command, args, volume_mount, working_dir, resource_memory, resource_cpu,
                       resource_gpu, image_pull_policy, image, env, privileged=False, username='', ports=None,
                       health=None,hostPort=[],resource_rdma=0):



        k8s_volumes, k8s_volume_mounts = self.get_volume_mounts(volume_mount, username)

        # 添加env
        env_list = []
        if env and type(env) == str:
            envs = re.split('\r|\n', env)
            # envs = [env.split('=') for env in envs if env and len(env.split('=')) == 2]
            envs = [[env[:env.index('=')], env[env.index('=') + 1:]] for env in envs if env and '=' in env]
            env_list = [client.V1EnvVar(name=env[0], value=env[1]) for env in envs]
        if env and type(env) == dict:
            env_list = [client.V1EnvVar(name=str(env_key), value=str(env[env_key])) for env_key in env]

        # 添加公共环境变量
        env_list.append(client.V1EnvVar(name='K8S_NODE_NAME', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='spec.nodeName'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_NAMESPACE', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='metadata.namespace'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_IP', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='status.podIP'))))
        env_list.append(client.V1EnvVar(name='K8S_HOST_IP', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='status.hostIP'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_NAME', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='metadata.name'))))

        security_context = client.V1SecurityContext(privileged=privileged) if privileged else None
        resources_requests = {}
        resources_limits = {}
        if resource_memory and not '~' in resource_memory:
            resource_memory = resource_memory.strip() + "~" + resource_memory.strip()
        if resource_cpu and not '~' in resource_cpu:
            resource_cpu = resource_cpu.strip() + "~" + resource_cpu.strip()

        if resource_memory:
            requests_memory, limits_memory = resource_memory.strip().split('~')
            resources_requests['memory'] = requests_memory.strip()
            resources_limits['memory'] = limits_memory.strip()

        if resource_cpu:
            requests_cpu, limits_cpu = resource_cpu.strip().split('~')
            resources_requests['cpu'] = requests_cpu.strip()
            resources_limits['cpu'] = limits_cpu.strip()

        gpu_num, gpu_type, resource_name = self.get_gpu(resource_gpu)

        # 整卡占用
        if gpu_num >= 1:
            gpu_num = int(gpu_num)
            if resource_name:
                resources_requests[resource_name] = str(int(gpu_num))
                resources_limits[resource_name] = str(int(gpu_num))

        DEFAULT_POD_RESOURCES = conf.get('DEFAULT_POD_RESOURCES',{})
        for resource_name in DEFAULT_POD_RESOURCES:
            if resource_name not in resources_limits:
                resources_requests[resource_name] = DEFAULT_POD_RESOURCES[resource_name]
                resources_limits[resource_name] = DEFAULT_POD_RESOURCES[resource_name]

        resources_obj = client.V1ResourceRequirements(requests=resources_requests, limits=resources_limits)

        if ports:
            if type(ports) == str:
                ports = [int(port) for port in ports.split(',')]
            # ports_k8s = [client.V1ContainerPort(name='port%s' % index, protocol='TCP', container_port=port) for index, port in enumerate(ports)] if ports else None

            ports_k8s = [client.V1ContainerPort(name='port%s' % str(port), protocol='TCP', container_port=port) for port in ports] if ports else None
        else:
            ports_k8s = []
            if hostPort:
                ports_k8s = [client.V1ContainerPort(name='port%s' % str(port), protocol='TCP', container_port=port, host_port=port) for port in hostPort]

        #         readinessProbe:
        #           failureThreshold: 2
        #           httpGet:
        #             path: /v1/models/resnet50/versions/2/metadata
        #             port: http
        #           initialDelaySeconds: 10
        #           periodSeconds: 10
        #           timeoutSeconds: 5

        # 端口检测或者脚本检测   8080:/health    shell:python /health.py
        if health:
            if health[0:health.index(":")] == 'shell':
                command = health.replace("shell:").split(' ')
                command = [c for c in command if c]
                readiness_probe = client.V1Probe(_exec=client.V1ExecAction(command=command),failure_threshold=1,period_seconds=60,timeout_seconds=30,initial_delay_seconds=60)
            else:
                port = health[0:health.index(":")]  # 健康检查的port
                path = health[health.index(":") + 1:]
                port_name = "port" + port
                # 端口只能用名称，不能用数字，而且要在里面定义
                if int(port) not in ports:
                    ports_k8s.append(client.V1ContainerPort(name=port_name, protocol='TCP', container_port=port))

                readiness_probe = client.V1Probe(http_get=client.V1HTTPGetAction(path=path,port=port_name),failure_threshold=1,period_seconds=60,timeout_seconds=30,initial_delay_seconds=60)

            # print(readiness_probe)

        container = client.V1Container(
            name=name,
            command=command,
            args=args,
            image=image,
            working_dir=working_dir if working_dir else None,
            image_pull_policy=image_pull_policy,
            volume_mounts=k8s_volume_mounts if k8s_volume_mounts else None,
            resources=resources_obj,
            env=env_list,
            security_context=security_context,
            ports=ports_k8s,
            readiness_probe=readiness_probe if health else None
        )



        return container

    # @pysnooper.snoop()
    def make_pod(self, namespace, name, labels, command, args, volume_mount, working_dir, node_selector,
                 resource_memory, resource_cpu, resource_gpu, image_pull_policy, image_pull_secrets, image, hostAliases,
                 env, privileged, accounts, username, ports=None, restart_policy='OnFailure',
                 scheduler_name='default-scheduler', node_name='', health=None, annotations={}, hostPort=[],resource_rdma=0):
        if not labels:
            labels={}
        if scheduler_name == 'kube-batch':
            annotations['scheduling.k8s.io/group-name'] = name


        image_pull_secrets = [client.V1LocalObjectReference(image_pull_secret) for image_pull_secret in image_pull_secrets]
        affinity = None
        nodeSelector = {}
        if node_selector and '=' in node_selector:
            nodeSelector = {}
            for selector in re.split(',|;|\n|\t', node_selector):
                selector = selector.strip()
                if selector:
                    nodeSelector[selector.strip().split('=')[0].strip()] = selector.strip().split('=')[1].strip()

        gpu_num, gpu_type, resource_name = self.get_gpu(resource_gpu)
        # 设置卡型
        if gpu_type and gpu_type.strip():
            nodeSelector['gpu-type'] = gpu_type
        # 独占模式，尽量聚集在一个，避免卡零碎
        if gpu_num >= 1:
            nodeSelector['gpu'] = 'true'
            labels['gpu']='true'
            # 优先选择gpu卡占用的地方，这样不容易造成卡的零碎化占用
            affinity = client.V1Affinity(
                node_affinity=None,
                pod_anti_affinity=None,
                pod_affinity=client.V1PodAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    client.V1WeightedPodAffinityTerm(
                        pod_affinity_term=client.V1PodAffinityTerm(
                            topology_key="kubernetes.io/hostname",
                            label_selector = client.V1LabelSelector(
                                match_labels={
                                    "gpu": 'true'
                                }
                            )
                        ),
                        weight=10)])
                )
        k8s_volumes, k8s_volume_mounts = self.get_volume_mounts(volume_mount, username)

        containers = [self.make_container(name=name,
                                          command=command,
                                          args=args,
                                          volume_mount=volume_mount,
                                          working_dir=working_dir,
                                          resource_memory=resource_memory,
                                          resource_cpu=resource_cpu,
                                          resource_gpu=resource_gpu,
                                          image_pull_policy=image_pull_policy,
                                          image=image,
                                          env=env,
                                          privileged=privileged,
                                          username=username,
                                          ports=ports,
                                          health=health,
                                          hostPort=hostPort,
                                          resource_rdma=resource_rdma
                                          )]

        # 添加host
        host_aliases = []
        if hostAliases:
            hostAliases_list = re.split('\r|\n', hostAliases)
            for row in hostAliases_list:
                hosts = row.strip().split(' ')
                hosts = [host.strip() for host in hosts if host.strip()]
                if len(hosts) > 1:
                    host_aliase = client.V1HostAlias(ip=hosts[0], hostnames=hosts[1:])
                    host_aliases.append(host_aliase)

        service_account = accounts if accounts else None
        spec = v1_pod_spec.V1PodSpec(affinity=affinity,image_pull_secrets=image_pull_secrets, node_selector=nodeSelector,node_name=node_name if node_name else None,
                                     volumes=k8s_volumes, containers=containers, restart_policy=restart_policy,
                                     host_aliases=host_aliases, service_account=service_account,scheduler_name=scheduler_name)
        metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels, annotations=annotations)
        pod = v1_pod.V1Pod(api_version='v1', kind='Pod', metadata=metadata, spec=spec)
        return pod, spec

    # @pysnooper.snoop()
    def create_debug_pod(self, namespace, name, labels, command, args, volume_mount, working_dir, node_selector,
                         resource_memory, resource_cpu, resource_gpu, image_pull_policy, image_pull_secrets, image,
                         hostAliases, env, privileged, accounts, username, scheduler_name='default-scheduler',
                         node_name='',annotations={},hostPort=[],resource_rdma=0):
        try:
            self.v1.delete_namespaced_pod(name=name, namespace=namespace, grace_period_seconds=0)
            # time.sleep(1)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
            pass
            # print(e)
        pod, pod_spec = self.make_pod(
            namespace=namespace,
            name=name,
            labels=labels,
            annotations=annotations,
            command=command,
            args=args,
            volume_mount=volume_mount,
            working_dir=working_dir,
            node_selector=node_selector,
            resource_memory=resource_memory,
            resource_cpu=resource_cpu,
            resource_gpu=resource_gpu,
            image_pull_policy=image_pull_policy,
            image_pull_secrets=image_pull_secrets,
            image=image,
            hostAliases=hostAliases,
            env=env,
            privileged=privileged,
            accounts=accounts,
            username=username,
            restart_policy='Never',
            scheduler_name=scheduler_name,
            node_name=node_name,
            hostPort=hostPort,
            resource_rdma=resource_rdma
        )
        # print(pod)
        pod = self.v1.create_namespaced_pod(namespace, pod)
        time.sleep(1)

    # 创建hubsecret
    # @pysnooper.snoop()
    def apply_hubsecret(self, namespace, name, user, password, server):
        try:
            hubsecrest = self.v1.read_namespaced_secret(name=name, namespace=namespace)
            if hubsecrest:
                self.v1.delete_namespaced_secret(name, namespace=namespace)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)

        cred_payload = {
            "auths": {
                server: {
                    "username": user,
                    "password": password,
                    "auth": base64.b64encode((user + ":" + password).encode()).decode(),
                }
            }
        }

        data = {
            ".dockerconfigjson": base64.b64encode(
                json.dumps(cred_payload).encode()
            ).decode()
        }
        secret = client.V1Secret(
            api_version="v1",
            data=data,
            kind="Secret",
            metadata=dict(name=name, namespace=namespace),
            type="kubernetes.io/dockerconfigjson",
        )

        secret_objects = self.v1.create_namespaced_secret(namespace=namespace, body=secret)
        return secret_objects

    # 创建notebook
    def create_crd(self,group,version,plural,namespace,body):
        crd_objects = client.CustomObjectsApi().create_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural,body=body)
        return crd_objects

    # 创建pod
    def create_pod(self, namespace, body):
        pod_object = self.v1.create_namespaced_pod(namespace=namespace, body=body)
        return pod_object

    #
    # def get_deployment(self,name,namespace):
    #     client.AppsV1Api().  (name, namespace)
    #     return []


    # @pysnooper.snoop(watch_explode=())
    def create_ReplicationController(self, namespace, name, replicas, labels, command, args, volume_mount, working_dir,
                          node_selector, resource_memory, resource_cpu, resource_gpu, image_pull_policy,
                          image_pull_secrets, image, hostAliases, env, privileged, accounts, username, ports,
                          scheduler_name='default-scheduler', health=None, annotations={},**kwargs):

        pod, pod_spec = self.make_pod(
            namespace=namespace,
            name=name,
            labels=labels,
            annotations=annotations,
            command=command,
            args=args,
            volume_mount=volume_mount,
            working_dir=working_dir,
            node_selector=node_selector,
            resource_memory=resource_memory,
            resource_cpu=resource_cpu,
            resource_gpu=resource_gpu,
            image_pull_policy=image_pull_policy,
            image_pull_secrets=image_pull_secrets,
            image=image,
            hostAliases=hostAliases,
            env=env,
            privileged=privileged,
            accounts=accounts,
            username=username,
            ports=ports,
            scheduler_name=scheduler_name,
            health=health
        )

        pod_spec.restart_policy = 'Always'  # dp里面必须是Always
        # cpu任务， # 控制pod尽量分散到不同的机器上
        if not resource_gpu or resource_gpu=='0':
            pod_spec.affinity = client.V1Affinity(
                pod_anti_affinity=client.V1PodAntiAffinity(
                    preferred_during_scheduling_ignored_during_execution=[client.V1WeightedPodAffinityTerm(
                        weight=10,
                        pod_affinity_term=client.V1PodAffinityTerm(
                            label_selector=client.V1LabelSelector(
                                match_expressions=[client.V1LabelSelectorRequirement(
                                    key=label[0],
                                    operator='In',
                                    values=[label[1]]
                                )]
                            ),
                            topology_key="kubernetes.io/hostname"
                        )

                    ) for label in labels.items()]
                ))

        return pod,pod_spec

    # 删除deployment
    def delete_deployment(self, namespace, name=None, labels=None):
        if name:
            try:
                client.AppsV1Api().delete_namespaced_deployment(name=name, namespace=namespace, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
        elif labels:
            try:
                labels_arr = ["%s=%s" % (key, labels[key]) for key in labels]
                labels_str = ','.join(labels_arr)
                deploys = self.AppsV1Api.list_namespaced_deployment(namespace=namespace, label_selector=labels_str).items
                for deploy in deploys:
                    client.AppsV1Api().delete_namespaced_deployment(name=deploy.metadata.name, namespace=namespace, grace_period_seconds=0)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)

    # deploymnet伸缩容
    def scale_deployment(self,namespace, name, replicas):
        try:
            deployment = self.AppsV1Api.read_namespaced_deployment(name=name, namespace=namespace)
            deployment.spec.replicas = int(replicas)
            self.AppsV1Api.replace_namespaced_deployment(name=name, namespace=namespace, body=deployment)
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def create_deployment(self, namespace, name, replicas, labels, command, args, volume_mount,working_dir,
                                     node_selector, resource_memory, resource_cpu, resource_gpu, image_pull_policy,
                                     image_pull_secrets, image, hostAliases, env, privileged, accounts, username,
                                     ports,
                                     scheduler_name='default-scheduler', health=None, annotations={}):
        pod,pod_spec = self.create_ReplicationController(
            namespace=namespace,name=name,replicas=replicas,labels=labels,command=command,args=args,volume_mount=volume_mount,
            working_dir=working_dir,node_selector=node_selector,resource_memory=resource_memory,resource_cpu=resource_cpu,
            resource_gpu=resource_gpu,image_pull_policy=image_pull_policy,image_pull_secrets=image_pull_secrets,image=image,
            hostAliases=hostAliases,env=env,privileged=privileged,accounts=accounts,username=username,ports=ports,scheduler_name=scheduler_name,
            health=health,annotations=annotations
        )
        metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels, annotations=annotations)
        selector = client.models.V1LabelSelector(match_labels=labels)
        template_metadata = v1_object_meta.V1ObjectMeta(labels=labels,annotations=annotations)
        template = client.models.V1PodTemplateSpec(metadata=template_metadata, spec=pod_spec)
        dp_spec = v1_deployment_spec.V1DeploymentSpec(replicas=int(replicas), selector=selector, template=template)
        dp = v1_deployment.V1Deployment(api_version='apps/v1', kind='Deployment', metadata=metadata, spec=dp_spec)
        # print(dp.to_str())
        # try:
        #     client.AppsV1Api().delete_namespaced_deployment(name, namespace)
        # except Exception as e:
        #     print(e)

        try:
            self.AppsV1Api.read_namespaced_deployment(name=name, namespace=namespace)
            # self.AppsV1Api.patch_namespaced_deployment(name=name, namespace=namespace, body=dp)
            self.AppsV1Api.replace_namespaced_deployment(name=name, namespace=namespace, body=dp)
        except ApiException as e:
            if e.status == 404:
                dp = self.AppsV1Api.create_namespaced_deployment(namespace, dp)

        # try:
        #     dp = client.AppsV1Api().create_namespaced_deployment(namespace, dp)
        # except Exception as e:
        #     print(e)
        #     try:
        #         client.AppsV1Api().patch_namespaced_deployment(name=name,namespace=namespace,body=dp)
        #     except Exception as e1:
        #         print(e1)
        # # time.sleep(2)

    # 删除statefulset
    # @pysnooper.snoop()
    def delete_statefulset(self, namespace, name=None, labels=None):
        if name:
            try:
                client.AppsV1Api().delete_namespaced_stateful_set(name=name, namespace=namespace)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)
        elif labels:
            try:
                labels_arr = ["%s=%s" % (key, labels[key]) for key in labels]
                labels_str = ','.join(labels_arr)
                stss = self.AppsV1Api.list_namespaced_stateful_set(namespace=namespace, label_selector=labels_str).items
                for sts in stss:
                    client.AppsV1Api().delete_namespaced_stateful_set(name=sts.metadata.name, namespace=namespace)
            except ApiException as api_e:
                if api_e.status != 404:
                    print(api_e)
            except Exception as e:
                print(e)

    # @pysnooper.snoop(watch_explode=())
    def create_statefulset(self, namespace, name, replicas, labels, command, args, volume_mount, working_dir,
                          node_selector, resource_memory, resource_cpu, resource_gpu, image_pull_policy,
                          image_pull_secrets, image, hostAliases, env, privileged, accounts, username, ports,
                          scheduler_name='default-scheduler', health=None, annotations={}):

        pod,pod_spec = self.create_ReplicationController(
            namespace=namespace, name=name, replicas=replicas, labels=labels, command=command, args=args,
            volume_mount=volume_mount,
            working_dir=working_dir, node_selector=node_selector, resource_memory=resource_memory,
            resource_cpu=resource_cpu,
            resource_gpu=resource_gpu, image_pull_policy=image_pull_policy, image_pull_secrets=image_pull_secrets,
            image=image,
            hostAliases=hostAliases, env=env, privileged=privileged, accounts=accounts, username=username, ports=ports,
            scheduler_name=scheduler_name,
            health=health, annotations=annotations
        )
        metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels)
        selector = client.models.V1LabelSelector(match_labels=labels)
        template_metadata = v1_object_meta.V1ObjectMeta(labels=labels)
        template = client.models.V1PodTemplateSpec(metadata=template_metadata, spec=pod_spec)
        sts_spec = client.models.V1StatefulSetSpec(pod_management_policy='OrderedReady',replicas=int(replicas), selector=selector,template=template,service_name=name)
        sts = client.models.V1StatefulSet(api_version='apps/v1', kind='StatefulSet', metadata=metadata, spec=sts_spec)
        # print(dp.to_str())

        try:
            self.AppsV1Api.read_namespaced_stateful_set(name=name, namespace=namespace)
            # self.AppsV1Api.patch_namespaced_deployment(name=name, namespace=namespace, body=dp)
            self.AppsV1Api.replace_namespaced_stateful_set(name=name, namespace=namespace, body=sts)
        except ApiException as e:
            if e.status == 404:
                dp = self.AppsV1Api.create_namespaced_stateful_set(namespace, sts)


    # 创建pod
    # @pysnooper.snoop()
    def create_service(self,namespace,name,username,ports,selector,service_type='ClusterIP',external_ip=None,annotations=None,load_balancer_ip=None,external_traffic_policy=None,disable_load_balancer=False):
        svc_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=selector,annotations=annotations)
        service_ports=[]
        for index,port in enumerate(ports):
            if type(port)==list and len(port)>1:
                service_ports.append(client.V1ServicePort(name='http%s'%index, node_port=int(port[0]) if service_type=='NodePort' else None, port=int(port[0]), protocol='TCP', target_port=int(port[1])))
            else:
                service_ports.append(client.V1ServicePort(name='http%s' % index, node_port=int(port) if service_type=='NodePort' else None, port=int(port), protocol='TCP', target_port=int(port)))

        svc_spec = client.V1ServiceSpec(cluster_ip='None' if disable_load_balancer else None, ports=service_ports,
                                        selector=selector, type=service_type, external_i_ps=external_ip,
                                        load_balancer_ip=load_balancer_ip,
                                        external_traffic_policy=external_traffic_policy)

        service = client.V1Service(api_version='v1', kind='Service', metadata=svc_metadata, spec=svc_spec)
        # print(service.to_dict())
        # try:
        #     self.v1.delete_namespaced_service(name, namespace)
        # except Exception as e:
        #     print(e)
        # try:
        #     service = self.v1.create_namespaced_service(namespace, service)
        # except Exception as e:
        #     print(e)

        try:
            self.v1.read_namespaced_service(name=name, namespace=namespace)
            self.v1.replace_namespaced_service(name=name, namespace=namespace, body=service)
        except ApiException as e:
            if e.status == 404:
                pass
                # print(service)
                service = self.v1.create_namespaced_service(namespace, body=service)

    # @pysnooper.snoop()
    def create_headless_service(self,namespace,name,username,run_id):
        svc_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels={"app":name,'user':username,"run-id":run_id})
        svc_spec = client.V1ServiceSpec(cluster_ip='None', selector={"app":name,'user':username},type='ClusterIP')
        service = client.V1Service(api_version='v1', kind='Service', metadata=svc_metadata, spec=svc_spec)
        # print(service.to_dict())
        try:
            self.v1.delete_namespaced_service(name, namespace)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            pass
            print(e)
        try:
            service = self.v1.create_namespaced_service(namespace, service)
        except Exception as e:
            print(e)

    # 创建pod
    # @pysnooper.snoop()
    def create_ingress(self, namespace, name, host, username, port):
        self.v1beta1 = client.ExtensionsV1beta1Api()
        ingress_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels={"app":name,'user':username},annotations={"nginx.ingress.kubernetes.io/proxy-connect-timeout":"3000","nginx.ingress.kubernetes.io/proxy-send-timeout":"3000","nginx.ingress.kubernetes.io/proxy-read-timeout":"3000","nginx.ingress.kubernetes.io/proxy-body-size":"1G"})
        backend = client.ExtensionsV1beta1IngressBackend(service_name=name,service_port=port)
        path = client.ExtensionsV1beta1HTTPIngressPath(backend=backend,path='/')
        http = client.ExtensionsV1beta1HTTPIngressRuleValue(paths=[path])
        rule = client.ExtensionsV1beta1IngressRule(host=host, http=http)
        ingress_spec = client.ExtensionsV1beta1IngressSpec(rules=[rule])
        ingress = client.ExtensionsV1beta1Ingress(api_version='extensions/v1beta1', kind='Ingress', metadata=ingress_metadata, spec=ingress_spec)
        # print(ingress.to_dict())
        try:
            self.v1beta1.delete_namespaced_ingress(name=name, namespace=namespace)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)

        try:
            ingress = self.v1beta1.create_namespaced_ingress(namespace=namespace, body=ingress)
        except Exception as e:
            print(e)

    #
    def delete_istio_ingress(self, namespace, name):
        crd_info = {
            "group": "networking.istio.io",
            "version": "v1alpha3",
            "plural": "virtualservices",
            'kind': 'VirtualService',
            "timeout": 60 * 60 * 24 * 1
        }
        try:
            self.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                            namespace=namespace, name=name)
        except Exception as e:
            print(e)

        try:
            self.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                            namespace=namespace, name=name+"-8080")
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def create_istio_ingress(self, namespace, name, host, ports, canary=None, shadow=None):
        crd_info = {
            "group": "networking.istio.io",
            "version": "v1alpha3",
            "plural": "virtualservices",
            'kind': 'VirtualService',
            "timeout": 60 * 60 * 24 * 1
        }

        crd_list = self.get_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace)
        # for vs_obj in crd_list:
        #     if vs_obj['name'] == name or vs_obj['name']== name+"-8080":
        #         self.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
        #                        namespace=namespace, name=vs_obj['name'])
        #         time.sleep(1)

        if len(ports) > 0:
            crd_json = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": name,
                    "namespace": namespace
                },
                "spec": {
                    "gateways": [
                        "kubeflow/kubeflow-gateway",
                    ],
                    "hosts": [
                        host
                    ],
                    "http": [
                        {
                            "route": [
                                {
                                    "destination": {
                                        "host": "%s.%s.svc.cluster.local" % (name, namespace),
                                        "port": {
                                            "number": int(ports[0])
                                        }
                                    }
                                }
                            ],
                            "timeout": "3000s"
                        }
                    ]
                }
            }

            def get_canary(gateway_service, canarys):
                canarys = re.split(',|;', canarys)
                des_canary = {}
                for canary in canarys:
                    service_name, traffic = canary.split(':')[0], canary.split(':')[1]
                    des_canary[service_name] = int(traffic.replace('%', ''))
                sum_traffic = sum(des_canary.values())
                gateway_service_traffic = 100 - sum_traffic
                if gateway_service_traffic > 0:
                    des_canary[gateway_service] = gateway_service_traffic
                    return des_canary
                else:
                    return {}

            # 添加分流配置
            if canary:
                canarys = get_canary(name, canary)
                if canarys:
                    route = []
                    for service_name in canarys:
                        destination = {
                            "destination": {
                                "host": "%s.%s.svc.cluster.local" % (service_name, namespace),
                                "port": {
                                    "number": int(ports[0])
                                }
                            },
                            "weight": int(canarys[service_name])
                        }
                        route.append(destination)

                    crd_json['spec']['http'][0]['route'] = route

            # 添加流量复制
            if shadow:
                shadow = re.split(',|;', shadow)[0]  # 只能添加一个流量复制
                service_name, traffic = shadow.split(':')[0], int(shadow.split(':')[1].replace("%", ''))

                mirror = {
                    "host": "%s.%s.svc.cluster.local" % (service_name, namespace),
                    "port": {
                        "number": int(ports[0])
                    }
                }
                mirror_percent = traffic

                crd_json['spec']['http'][0]['mirror'] = mirror
                crd_json['spec']['http'][0]['mirror_percent'] = mirror_percent

            try:
                client.CustomObjectsApi().get_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    plural=crd_info['plural'],
                    name=name,
                    namespace=namespace
                )
                crd_objects = client.CustomObjectsApi().replace_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    namespace=namespace,
                    plural=crd_info['plural'],
                    name=name,
                    body=crd_json
                )
            except ApiException as e:
                if e.status == 404:
                    crd_objects = client.CustomObjectsApi().create_namespaced_custom_object(
                        group=crd_info['group'],
                        version=crd_info['version'],
                        namespace=namespace,
                        plural=crd_info['plural'],
                        body=crd_json)

        if len(ports) > 1:
            crd_json = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": name + "-8080",
                    "namespace": namespace
                },
                "spec": {
                    "gateways": [
                        "kubeflow/kubeflow-gateway-8080",
                    ],
                    "hosts": [
                        host
                    ],
                    "http": [
                        {
                            "route": [
                                {
                                    "destination": {
                                        "host": "%s.service.svc.cluster.local" % name,
                                        "port": {
                                            "number": int(ports[1])
                                        }
                                    }
                                }
                            ],
                            "timeout": "3000s"
                        }
                    ]
                }
            }

            try:
                client.CustomObjectsApi().get_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    plural=crd_info['plural'],
                    name=name + '-8080',
                    namespace=namespace
                )
                crd_objects = client.CustomObjectsApi().replace_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    namespace=namespace,
                    plural=crd_info['plural'],
                    name=name + '-8080',
                    body=crd_json
                )
            except ApiException as e:
                if e.status == 404:
                    crd_objects = client.CustomObjectsApi().create_namespaced_custom_object(
                        group=crd_info['group'],
                        version=crd_info['version'],
                        namespace=namespace,
                        plural=crd_info['plural'],
                        body=crd_json)

    def delete_volcano(self, namespace, name):
        crd_info = {
            "group": "batch.volcano.sh",
            "version": "v1alpha1",
            "plural": "jobs",
            'kind': 'Job',
            "timeout": 60 * 60 * 24 * 1
        }
        try:
            self.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=name)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)

    def delete_configmap(self, namespace, name):
        try:
            self.v1.delete_namespaced_config_map(name=name, namespace=namespace, grace_period_seconds=0)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def create_configmap(self, namespace, name, data, labels):
        try:
            self.v1.delete_namespaced_config_map(name=name, namespace=namespace)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
        try:
            meta = client.V1ObjectMeta(name=name, labels=labels)
            configmap = client.V1ConfigMap(data=data, metadata=meta)
            self.v1.create_namespaced_config_map(namespace=namespace, body=configmap)
        except Exception as e:
            print(e)

    def delete_hpa(self, namespace, name):
        try:
            client.AutoscalingV2beta1Api().delete_namespaced_horizontal_pod_autoscaler(name=name,namespace=namespace,grace_period_seconds=0)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)
        try:
            client.AutoscalingV1Api().delete_namespaced_horizontal_pod_autoscaler(name=name,namespace=namespace,grace_period_seconds=0)
        except ApiException as api_e:
            if api_e.status != 404:
                print(api_e)
        except Exception as e:
            print(e)

    # @pysnooper.snoop()
    def create_hpa(self,namespace,name,min_replicas,max_replicas,hpa):
        self.delete_hpa(namespace,name)
        hpa = re.split(',|;', hpa)

        hpa_json = {
            "apiVersion": "autoscaling/v2beta1",  # 需要所使用的k8s集群启动了这个版本的hpa，可以通过 kubectl api-resources  查看使用的版本
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": name
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [

                ]
            }
        }

        for threshold in hpa:
            if 'mem' in threshold:
                mem_threshold = re.split(':|=', threshold)[1].replace('%', '')
                hpa_json['spec']['metrics'].append(
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "targetAverageUtilization": int(mem_threshold),  # V1的书写格式
                            # "target": {       # V2 的书写格式
                            #     "type": "Utilization",
                            #     "averageUtilization": int(mem_threshold)
                            # }
                        }
                    }
                )

            if 'cpu' in threshold:
                cpu_threshold = re.split(':|=', threshold)[1].replace('%', '')
                hpa_json['spec']['metrics'].append(
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "targetAverageUtilization": int(cpu_threshold),  # V1的书写格式
                            # "target": {       # V2 的书写格式
                            #     "type": "Utilization",
                            #     "averageUtilization": int(cpu_threshold)
                            # }
                        }
                    }
                )

            if 'gpu' in threshold:
                gpu_threshold = re.split(':|=', threshold)[1].replace('%', '')
                hpa_json['spec']['metrics'].append(
                    {
                        "type": "Pods",
                        "pods": {
                            "metricName": "container_gpu_usage",
                            "targetAverageValue": int(gpu_threshold) / 100
                        }
                    }
                )

        # my_conditions.append(client.V2beta1HorizontalPodAutoscalerCondition(status="True", type='AbleToScale'))
        #
        # status = client.V2beta1HorizontalPodAutoscalerStatus(conditions=my_conditions, current_replicas=max_replicas,
        #                                                      desired_replicas=max_replicas)
        # # 自定义指标进行hpa，需要在autoscaling/v2beta1下面
        # body = client.V2beta1HorizontalPodAutoscaler(
        #     api_version='autoscaling/v2beta1',
        #     kind='HorizontalPodAutoscaler',
        #     metadata=client.V1ObjectMeta(name=name),
        #     spec=client.V2beta1HorizontalPodAutoscalerSpec(
        #         max_replicas=max_replicas,
        #         min_replicas=min_replicas,
        #         metrics=my_metrics,
        #         scale_target_ref=client.V2beta1CrossVersionObjectReference(kind='Deployment', name=name,
        #                                                                    api_version='apps/v1'),
        #     ),
        #     status=status
        # )
        print(json.dumps(hpa_json, indent=4, ensure_ascii=4))
        try:
            client.AutoscalingV2beta1Api().create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa_json, pretty=True)
        except ValueError as e:
            if str(e) == 'Invalid value for `conditions`, must not be `None`':
                print(e)
            else:
                print(e)
                raise e

    # @pysnooper.snoop()
    def to_memory_GB(self, memory):
        if 'K' in memory:
            return round(float(memory.replace('Ki', '').replace('K', '')) / 1000 / 1000,2)
        if 'M' in memory:
            return round(float(memory.replace('Mi', '').replace('M', '')) / 1000,2)
        if 'G' in memory:
            return round(float(memory.replace('Gi', '').replace('G', '')),2)
        return 0

    def to_cpu(self, cpu):
        if 'm' in cpu:
            return round(float(cpu.replace('m', '')) / 1000,2)
        if 'n' in cpu:
            return round(float(cpu.replace('n', '')) / 1000 / 1000,2)
        if 'u' in cpu:
            return round(float(cpu.replace('u', '')) / 1000 / 1000 / 1000,2)
        return round(float(cpu),2)

    # @pysnooper.snoop(watch_explode=('item'))
    def get_node_metrics(self):
        back_metrics = []
        cust = client.CustomObjectsApi()
        metrics = cust.list_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'nodes')  # All node metrics
        items = metrics['items']
        for item in items:
            back_metrics.append({
                "name": item['metadata']['name'],
                "time": item['timestamp'],
                "cpu": self.to_cpu(item['usage']['cpu']),
                "memory": self.to_memory_GB(item['usage']['memory']),
                "window": item['window'],
            })
        # print(back_metrics)
        return back_metrics

    # @pysnooper.snoop()
    def get_pod_metrics(self, namespace=None):
        back_metrics = []
        cust = client.CustomObjectsApi()
        if namespace:
            metrics = cust.list_namespaced_custom_object('metrics.k8s.io', 'v1beta1', namespace,'pods')  # Just pod metrics for the default namespace
        else:
            metrics = cust.list_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'pods')  # All Pod Metrics
        items = metrics.get('items', [])
        # print(items)
        for item in items:
            try:
                back_metrics.append({
                    "name": item['metadata']['name'],
                    "time": item['timestamp'],
                    "namespace": item['metadata']['namespace'],
                    "cpu": sum(self.to_cpu(container['usage']['cpu']) for container in item['containers']),
                    "memory": sum(self.to_memory_GB(container['usage']['memory']) for container in item['containers']),
                    "window": item['window']
                })
            except Exception as e:
                traceback.extract_stack()
                print(e)
        # print(back_metrics)
        return back_metrics

    # @pysnooper.snoop()
    def exec_command(self, name, namespace, command):
        try:
            self.v1.read_namespaced_pod(name=name, namespace=namespace)
        except ApiException as e:
            if e.status != 404:
                print("Unknown error: %s" % e)
                return

        self.v1.connect_get_namespaced_pod_exec(
            name,
            namespace,
            command=command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False
        )

    # 实时跟踪指定pod日志，直到pod结束
    def get_pod_log_stream(self, name, namespace, container, tail_lines=100):
        """
        获取pod的日志
        :param tail_lines: # 显示最后多少行
        :return:
        """
        try:
            # stream pod log
            streams = self.v1.read_namespaced_pod_log(
                name,
                namespace,
                container=container,
                follow=True,
                _preload_content=False,
                timestamps=False,
                tail_lines=tail_lines).stream()
            return streams

        except ApiException as e:
            if e.status == 404:
                print("Get Log not fund Podname: {0}".format(name))
                raise Exception("获取日志时，未找到此pod: {0}".format(name))
            if e.status == 400:
                raise Exception("容器并未创建成功，请联系运维人员进行排查。")
            raise e
        except Exception as e:
            print("Get Log Fail: {0}".format(str(e)))
            raise e

    def download_pod_log(self, name, namespace, container=None, tail_lines=None, since_seconds=None, since_time=None):
        print('begin donwload log')
        logs = self.v1.read_namespaced_pod_log(
            name=name,
            namespace=namespace,
            container=container,
            pretty=True,
            _preload_content=True,
            tail_lines=int(tail_lines) if tail_lines else None,
            since_seconds = int(since_seconds) if since_seconds else (datetime.datetime.now().timestamp()-int(since_time)) if since_time else None,
            # timestamps=True
        )
        return logs

    def get_uesd_gpu(self, namespaces):
        all_gpu_pods = []

        def get_used_gpu(pod):
            name = pod.metadata.name
            user = ''
            if pod.metadata.labels:
                user = pod.metadata.labels.get('run-rtx', '')
                if not user:
                    user = pod.metadata.labels.get('user', '')
                if not user:
                    user = pod.metadata.labels.get('rtx-user', '')

            containers = pod.spec.containers

            gpu = 0
            for container in containers:
                for gpu_resource_name in list(self.gpu_resource.values()):
                    limits = container.resources.limits
                    request = container.resources.requests
                    container_gpu = 0
                    if limits:
                        container_gpu = int(limits.get(gpu_resource_name, 0))
                    elif request:
                        container_gpu = int(request.get(gpu_resource_name, 0))
                    if container_gpu < 0.01:
                        container_gpu = 0
                    gpu += container_gpu
            return name, user, gpu

        for namespace in namespaces:
            pods = self.v1.list_namespaced_pod(namespace).items
            for pod in pods:
                status = pod.status.phase
                if status != 'Running':
                    continue
                name, user, gpu_num = get_used_gpu(pod)
                if gpu_num:
                    all_gpu_pods.append({
                        "name": name,
                        "user": user,
                        "gpu": gpu_num,
                        "namespace": namespace
                    })

        return all_gpu_pods

    def make_sidecar(self, agent_name):
        if agent_name.upper() == 'L5':
            pass
        pass

    def to_local_time(self,time_str):
        if not time_str:
            return time_str
        if type(time_str)==str:
            return (datetime.datetime.strptime(time_str.replace('T', ' ').replace('Z', ''),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
        elif type(time_str)==datetime.datetime:
            return (time_str+datetime.timedelta(hours=8)).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S')

    def terminal_start(self, namespace, pod_name, container, cols=80, rows=24):
        command = [
            "/bin/sh",
            "-c",
            'TERM=xterm-256color; export TERM; [ -x /bin/bash ] '
            '&& ([ -x /usr/bin/script ] '
            '&& /usr/bin/script -q -c "/bin/bash" /dev/null || exec /bin/bash) '
            '|| exec /bin/sh']

        container_stream = stream(
            self.v1.connect_get_namespaced_pod_exec,
            name=pod_name,
            namespace=namespace,
            container=container,
            command=command,
            stderr=True, stdin=True,
            stdout=True, tty=True,
            _preload_content=False
        )
        container_stream.write_channel(4, json.dumps({"Height": int(rows), "Width": int(cols)}))
        return container_stream
    # 读取pvc
    def get_pvc(self,name,namespace):
        try:
            pvc = self.v1.read_namespaced_persistent_volume_claim(name,namespace)
            pvc = {
                "status":pvc.status.phase if pvc.status and pvc.status.phase else 'unknown'
            }
            return pvc
        except ApiException as e1:
            if e1.status == 404:
                pass

        except Exception  as e:
            pass
        return {}

        pass
class K8SStreamThread(threading.Thread):

    def __init__(self, ws, container_stream):
        super(K8SStreamThread, self).__init__()
        self.ws = ws
        self.stream = container_stream

    def run(self):
        while not self.ws.closed:

            if not self.stream.is_open():
                logging.info('container stream closed')
                self.ws.close()

            try:
                if self.stream.peek_stdout():
                    stdout = self.stream.read_stdout()
                    self.ws.send(stdout)

                if self.stream.peek_stderr():
                    stderr = self.stream.read_stderr()
                    self.ws.send(stderr)
            except Exception as err:
                logging.error('container stream err: {}'.format(err))
                self.ws.close()
                break


# @pysnooper.snoop()
def check_status_time(status, hour=8):
    if type(status) == dict:
        for key in status:
            try:
                if key=='startedAt' or key=='finishedAt':
                    if type(status[key])==datetime.datetime:
                        status[key]=status[key]-datetime.timedelta(hours=hour)
                    elif type(status[key])==str:
                        status[key] = (datetime.datetime.strptime(status[key].replace('T',' ').replace('Z',''),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(e)
            status[key] = check_status_time(status[key], hour)

    elif type(status) == list:
        for index in range(len(status)):
            status[index] = check_status_time(status[index], hour)

    return status


if __name__=='__main__':
    k8s_client = K8s(file_path='kubeconfig/dev-kubeconfig')

    print(k8s_client.get_node())


