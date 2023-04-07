import time,datetime,logging,os,sys

dir_common = os.path.split(os.path.realpath(__file__))[0] + '/../'
sys.path.append(dir_common)   # 将根目录添加到系统目录,才能正常引用common文件夹
import re
from kubernetes import client,config,watch
from kubernetes.client.models import v1_pod,v1_object_meta,v1_pod_spec,apps_v1beta1_deployment,apps_v1beta1_deployment_spec
from kubernetes.client.models import v1_stateful_set,v1_stateful_set_spec
import kubernetes
import yaml
from os import path
import json
# from common.config import *
import pysnooper
import multiprocessing
import random
from multiprocessing import Manager
import base64
import uuid
from kubernetes import config
from kubernetes.client.rest import ApiException


# K8s操作类型
class K8s():

    def __init__(self,file_path=None):  # kubeconfig
        kubeconfig = os.getenv('KUBECONFIG','')
        if file_path:
            config.kube_config.load_kube_config(config_file=file_path)
        elif kubeconfig:
            config.kube_config.load_kube_config(config_file=kubeconfig)
        else:
            config.load_incluster_config()   # 使用为pod配置的rbac访问集群
        self.v1 = client.CoreV1Api()
        self.v1beta1 = client.ExtensionsV1beta1Api()
        self.AppsV1Api = client.AppsV1Api()
        self.v1.api_client.configuration.verify_ssl = False  # 只能设置 /usr/local/lib/python3.6/dist-packages/kubernetes/client/configuration.py:   self.verify_ssl= True ---> False

    # 获取指定范围的pod
    # @pysnooper.snoop(watch_explode=('metadata',))
    def get_pods(self,namespace=None,service_name=None,pod_name=None,labels={}):
        try:
            all_pods =[]
            back_pods=[]
            # 如果只有命名空间
            if(namespace and not service_name and not pod_name and not labels):
                all_pods = self.v1.list_namespaced_pod(namespace).items
            # 如果有命名空间和pod名，就直接查询pod
            elif (namespace and pod_name):
                pod = self.v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                all_pods.append(pod)
            # 如果只有命名空间和服务名，就查服务下绑定的pod
            elif(namespace and service_name):  # 如果有命名空间和服务名
                all_endpoints = self.v1.read_namespaced_endpoints(service_name, namespace)  # 先查询入口点，
                subsets = all_endpoints.subsets
                addresses = subsets[0].addresses  # 只取第一个子网
                for address in addresses:
                    pod_name_temp = address.target_ref.name
                    pod = self.v1.read_namespaced_pod(name=pod_name_temp,namespace=namespace)
                    all_pods.append(pod)
            elif(namespace and labels):
                src_pods = self.v1.list_namespaced_pod(namespace).items
                for pod in src_pods:
                    pod_labels = pod.metadata.labels
                    is_des_pod = True
                    for key in labels:
                        if key not in pod_labels or pod_labels[key]!=labels[key]:
                            is_des_pod = False
                            break
                    if is_des_pod:
                        all_pods.append(pod)


            for pod in all_pods:
                # print(pod.status)
                metadata = pod.metadata
                status = pod.status.phase if pod and hasattr(pod,'status') and hasattr(pod.status,'phase') else ''
                temp={
                    'name':metadata.name,
                    'host_ip':pod.status.host_ip,
                    'pod_ip':pod.status.pod_ip,
                    'status':status,   # 每个容器都正常才算正常
                    'node_name':pod.spec.node_name,
                    "labels":metadata.labels,
                    "start_time":metadata.creation_timestamp+datetime.timedelta(hours=8)   # 时间格式
                }
                back_pods.append(temp)
            # print(back_pods)
            return back_pods

        except Exception as e:
            # print(e)
            return []


    # 获取 指定服务，指定命名空间的下面的endpoint
    def get_pod_ip(self,namespace='cloudai-2',service_name='face-search-vip-service'):
        try:
            all_pods = self.get_pods(namespace=namespace,service_name=service_name)
            all_pod_ip=[]
            if(all_pods):
                for pod in all_pods:
                    all_pod_ip.append(pod['pod_ip'])
                # print(all_pod_ip)
            return all_pod_ip
        except Exception as e:
            print(e)
            return None


    # 指定命名空间，指定服务名，指定pod名称，指定状态，删除重启pod。status为运行状态,True  或者False
    def delete_pods(self,namespace=None,service_name=None,pod_name=None,status=None,labels=None):
        allresponse = []
        if namespace and pod_name:
            api_response = self.v1.delete_namespaced_pod(name = pod_name, namespace=namespace,grace_period_seconds=0)
            allresponse.append(api_response)
            return allresponse
        try:
            if labels:
                all_pod = self.get_pods(namespace=namespace,labels=labels)
                for pod in all_pod:
                    api_response = self.v1.delete_namespaced_pod(pod['name'], namespace,grace_period_seconds=0)
                    allresponse.append(api_response)
            if status:
                all_pod = self.get_pods(namespace=namespace,service_name=service_name,pod_name=pod_name)
                if all_pod:
                    for pod in all_pod:
                        if status==None:   # 如果没有指定运行状态，则直接删除
                            api_response = self.v1.delete_namespaced_pod(pod['name'],namespace,grace_period_seconds=0)
                            allresponse.append(api_response)
                        elif pod['status']==status:
                            # body = kubernetes.client.V1DeleteOptions(grace_period_seconds=0,orphan_dependents=False)  # 不正常的要设置强制删除
                            api_response = self.v1.delete_namespaced_pod(pod['name'], namespace,grace_period_seconds=0)
                            allresponse.append(api_response)
                        print('delete pod %s' % all_pod)
            return allresponse
        except Exception as e:
            print(e)
            return []
    # 获取指定label的nodeip列表
    def get_node_ip(self,label):
        try:
            all_node_ip=[]
            all_node = self.v1.list_node(label_selector=label).items
            # print(all_node)
            for node in all_node:
                adresses=node.status.addresses
                for address in adresses:
                    if address.type=='InternalIP':
                        all_node_ip.append(address.address)
            return all_node_ip
        except Exception as e:
            print(e)
            return None

    # 获取指定label的nodeip列表
    def label_node(self,ips, label):
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
                            "labels": {
                                label: "true"
                            }
                        }
                    }
                    api_response = self.v1.patch_node(Hostname, body)

            return all_node_ip
        except Exception as e:
            print(e)
            return None

    # 根据各种crd自定义的status结构，判断最终评定的status
    # @pysnooper.snoop()
    def get_crd_status(self,crd_object,group,plural):
        status = ''
        # workflows 使用最后一个node的状态为真是状态
        if plural == 'workflows':
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
                    if condition['type']=='Ready' and condition['status']=='True':
                        status='ready'
        elif plural == 'jobs' and group=='batch.volcano.sh':
            status = 'unready'
            if 'status' in crd_object and 'state' in crd_object['status'] and 'phase' in  crd_object['status']['state']:
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
            self.crd = client.CustomObjectsApi()
            crd_object = self.crd.get_namespaced_custom_object(group=group, version=version, namespace=namespace,
                                                               plural=plural, name=name)
            ya = yaml.load(json.dumps(crd_object))
            ya_str = yaml.safe_dump(ya,default_flow_style=False)
            return ya_str
        except Exception as e:
            print(e)
        return ''


    # @pysnooper.snoop(watch_explode=('crd_object'))
    def get_one_crd(self, group, version, plural, namespace, name):
        try:
            self.crd = client.CustomObjectsApi()
            crd_object = self.crd.get_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural,name=name)
            if not crd_object:
                return {}

            # print(crd_object['status']['conditions'][-1]['type'])
            status = self.get_crd_status(crd_object,group,plural)

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            back_object = {
                "name": crd_object['metadata']['name'],
                "namespace": crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations": json.dumps(crd_object['metadata']['annotations'], indent=4,
                                          ensure_ascii=False) if 'annotations' in crd_object['metadata'] else '',
                "labels": json.dumps(crd_object['metadata']['labels'], indent=4, ensure_ascii=False) if 'labels' in
                                                                                                        crd_object[
                                                                                                            'metadata'] else '',
                "spec": json.dumps(crd_object['spec'], indent=4, ensure_ascii=False),
                "create_time": creat_time,
                "status": status,
                "status_more": json.dumps(crd_object['status'], indent=4,
                                          ensure_ascii=False) if 'status' in crd_object else ''
            }

                # return
            return back_object
        except Exception as e:
            print(e)
            return {}


    # @pysnooper.snoop(watch_explode=())
    def get_crd(self,group,version,plural,namespace,return_dict=None):
        self.crd = client.CustomObjectsApi()
        crd_objects = self.crd.list_namespaced_custom_object(group=group,version=version,namespace=namespace,plural=plural)['items']
        back_objects=[]
        for crd_object in crd_objects:
            # print(crd_object['status']['conditions'][-1]['type'])
            status = self.get_crd_status(crd_object,group,plural)

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            finish_time=''
            if 'status' in crd_object and 'finishedAt' in crd_object['status'] and crd_object['status']['finishedAt']:
                finish_time = crd_object['status']['finishedAt'].replace('T', ' ').replace('Z', '')
                finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            elif 'status' in crd_object and 'completionTime' in crd_object['status'] and crd_object['status']['completionTime']:
                finish_time = crd_object['status']['completionTime'].replace('T', ' ').replace('Z', '')
                finish_time = (datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            back_object={
                "name":crd_object['metadata']['name'],
                "namespace":crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations":json.dumps(crd_object['metadata']['annotations'],indent=4,ensure_ascii=False) if 'annotations' in crd_object['metadata'] else '',
                "labels": json.dumps(crd_object['metadata']['labels'], indent=4, ensure_ascii=False) if 'labels' in crd_object['metadata'] else '{}',
                "spec": json.dumps(crd_object['spec'], indent=4, ensure_ascii=False),
                "create_time":creat_time,
                "finish_time":finish_time,
                "status":status,
                "status_more":json.dumps(crd_object['status'],indent=4,ensure_ascii=False) if 'status' in crd_object else ''
            }
            back_objects.append(back_object)
            # return
        if return_dict!=None:
            return_dict[namespace]=back_objects
        return back_objects


    # @pysnooper.snoop(watch_explode=())
    def get_crd_all_namespaces(self,group,version,plural,pool=False):
        self.crd = client.CustomObjectsApi()
        all_namespace = self.v1.list_namespace().items
        all_namespace=[namespace.metadata.name for namespace in all_namespace]
        back_objects = []
        jobs = []
        if pool:
            from multiprocessing import Manager
            manager = Manager()
            return_dict = manager.dict()
            for namespace in all_namespace:
                p = multiprocessing.Process(target=self.get_crd, args=(group,version,plural,namespace,return_dict))
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
                crds = self.get_crd(group,version,plural,namespace)
                for crd_object in crds:
                    back_objects.append(crd_object)
            return back_objects



    # @pysnooper.snoop(watch_explode=())
    def delete_crd(self,group,version,plural,namespace,name='',labels=None):
        if name:
            try:
                self.crd = client.CustomObjectsApi()
                delete_body = client.V1DeleteOptions(grace_period_seconds=0)
                self.crd.delete_namespaced_custom_object(group=group,version=version,namespace=namespace,plural=plural,name=name,body=delete_body)
            except Exception as e:
                print(e)
            return [name]
        elif labels:
            back_name=[]
            crds = self.get_crd(group = group,version = version,plural = plural,namespace = namespace)
            for crd in crds:
                if crd['labels']:
                    crd_labels = json.loads(crd['labels'])
                    for key in labels:
                        if key in crd_labels and labels[key]==crd_labels[key]:
                            try:
                                self.crd = client.CustomObjectsApi()
                                delete_body = client.V1DeleteOptions(grace_period_seconds=0)
                                self.crd.delete_namespaced_custom_object(group=group, version=version, namespace=namespace,plural=plural, name=crd['name'], body=delete_body)
                            except Exception as e:
                                print(e)
                            back_name.append(crd['name'])
            return back_name

    # @pysnooper.snoop()
    def delete_workflow(self,all_crd_info,namespace,run_id):
        if not run_id:
            return None


        if run_id:

            # 删除workflow
            crd_info = all_crd_info['workflow']
            try:
                crd_names = self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'],
                    plural=crd_info['plural'], namespace=namespace, labels={'run-id': run_id}
                )

            except Exception as e:
                print(e)


            # 删除tfjob
            try:
                crd_info = all_crd_info['tfjob']
                crd_names = self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'],
                    namespace=namespace,labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)

            # 删除framework
            try:
                crd_info = all_crd_info['framework']
                crd_names = self.delete_crd(group=crd_info['group'], version=crd_info['version'],
                                                  plural=crd_info['plural'], namespace=namespace,
                                                  labels={"run-id": str(run_id)})
            except Exception as e:
                print(e)


            # 删除pytorchjob
            try:
                crd_info = all_crd_info['pytorchjob']
                crd_names = self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)

            # 删除mpijob
            try:
                crd_info = all_crd_info['mpijob']
                crd_names = self.delete_crd(
                    group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                    namespace=namespace, labels={'run-id': run_id}
                )
            except Exception as e:
                print(e)

            # 删除framework
            try:
                crd_info = all_crd_info['framework']
                crd_names = self.delete_crd(group=crd_info['group'], version=crd_info['version'],
                                                  plural=crd_info['plural'], namespace=namespace,
                                                  labels={"run-id": str(run_id)})
            except Exception as e:
                print(e)


            # 删除deployment
            try:
                self.delete_deployment(namespace=namespace, labels={'run-id': run_id})
            except Exception as e:
                print(e)

            # 删除stss
            try:
                stss = self.AppsV1Api.list_namespaced_stateful_set(namespace=namespace,label_selector="run-id=%s" % str(run_id)).items
                if stss:
                    for sts in stss:
                        self.AppsV1Api.delete_namespaced_stateful_set(namespace=namespace,name=sts.metadata.name,grace_period_seconds=0)

            except Exception as e:
                print(e)


            # 删除daemonsets
            try:
                daemonsets = self.AppsV1Api.list_namespaced_daemon_set(namespace=namespace,label_selector="run-id=%s" % str(run_id)).items
                if daemonsets:
                    for daemonset in daemonsets:
                        self.AppsV1Api.delete_namespaced_daemon_set(namespace=namespace,name=daemonset.metadata.name,grace_period_seconds=0)

            except Exception as e:
                print(e)

            # 删除sts
            try:
                stss = self.AppsV1Api.list_namespaced_stateful_set(namespace=namespace,label_selector="run-id=%s" % str(run_id)).items
                if stss:
                    for sts in stss:
                        self.AppsV1Api.delete_namespaced_stateful_set(namespace=namespace,name=sts.metadata.name,grace_period_seconds=0)

            except Exception as e:
                print(e)

            # 删除service
            try:
                services = self.v1.list_namespaced_service(namespace=namespace,label_selector="run-id=%s" % str(run_id)).items
                if services:
                    for service in services:
                        self.v1.delete_namespaced_service(namespace=namespace,name=service.metadata.name,grace_period_seconds=0)

            except Exception as e:
                print(e)

            # 不能删除pod，因为task的模板也是有这个run-id的，所以不能删除


    def delete_service(self,namespace,name):
        try:
            self.v1.delete_namespaced_service(name=name,namespace=namespace)
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
    #             volume_mounts_temp = re.split(',|;', volume_mount)
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
    def get_volume_mounts(self,volume_mount,username):
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
                        volumn_name = pvc_name.replace('_', '-').lower()[:60].lower().strip('-')
                        k8s_volumes.append({
                            "name":volumn_name,
                            "persistentVolumeClaim":{
                                "claimName":pvc_name
                            }
                        })
                        k8s_volume_mounts.append(
                            {
                                "name":volumn_name,
                                "mountPath":os.path.join(mount, username),
                                "subPath":username
                            }
                        )

                    if "(hostpath)" in volume:
                        hostpath_name = volume.replace('(hostpath)', '').replace(' ', '')
                        temps = re.split('_|\.|/', hostpath_name)
                        temps = [temp for temp in temps if temp]
                        volumn_name = '-'.join(temps).lower()[:60].lower().strip('-')
                        k8s_volumes.append(
                            {
                                "name":volumn_name,
                                "hostPath":{
                                    "path":hostpath_name
                                }
                            }
                        )
                        k8s_volume_mounts.append({
                            "name":volumn_name,
                            "mountPath":mount
                        })

                    if "(configmap)" in volume:
                        configmap_name = volume.replace('(configmap)', '').replace(' ', '')
                        volumn_name = configmap_name.replace('_', '-').replace('/', '-').replace('.', '-').lower()[:60].lower().strip('-')
                        k8s_volumes.append({
                            "name":volumn_name,
                            "configMap":{
                                "name":configmap_name
                            }
                        })

                        k8s_volume_mounts.append({
                            "name":volumn_name,
                            "mountPath":mount
                        })

                    if "(memory)" in volume:
                        memory_size = volume.replace('(memory)', '').replace(' ', '').lower().replace('g','')
                        volumn_name = ('memory-%s'%memory_size)[:60].lower().strip('-')
                        k8s_volumes.append({
                            "name":volumn_name,
                            "emptyDir":{
                                "medium": "Memory",
                                "sizeLimit": "%sGi"%memory_size
                            }
                        })

                        k8s_volume_mounts.append({
                            "name":volumn_name,
                            "mountPath":mount
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
        return k8s_volumes,k8s_volume_mounts


    # @pysnooper.snoop(watch_explode=('envs'))
    def make_container(self,name,command,args,volume_mount,working_dir,resource_memory,resource_cpu,resource_gpu,image_pull_policy,image,env,privileged=False,username='',ports=None):

        if not '~' in resource_memory:
            resource_memory = resource_memory.strip() + "~" + resource_memory.strip()
        if not '~' in resource_cpu:
            resource_cpu = resource_cpu.strip() + "~" + resource_cpu.strip()

        requests_memory, limits_memory = resource_memory.strip().split('~')
        requests_cpu, limits_cpu = resource_cpu.strip().split('~')

        k8s_volumes, k8s_volume_mounts = self.get_volume_mounts(volume_mount,username)

        # 添加env
        env_list = []
        if env and type(env)==str:
            envs = re.split('\r|\n', env)
            # envs = [env.split('=') for env in envs if env and len(env.split('=')) == 2]
            envs = [[env[:env.index('=')],env[env.index('=')+1:]] for env in envs if env and '=' in env]
            env_list = [client.V1EnvVar(name=env[0], value=env[1]) for env in envs]
        if env and type(env)==dict:
            env_list = [client.V1EnvVar(name=str(env_key), value=str(env[env_key])) for env_key in env]

        # 添加公共环境变量
        env_list.append(client.V1EnvVar(name='K8S_NODE_NAME', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='spec.nodeName'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_NAMESPACE', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='metadata.namespace'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_IP', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='status.podIP'))))
        env_list.append(client.V1EnvVar(name='K8S_HOST_IP', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='status.hostIP'))))
        env_list.append(client.V1EnvVar(name='K8S_POD_NAME', value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path='metadata.name'))))

        security_context = client.V1SecurityContext(privileged=privileged) if privileged else None
        gpu_type = os.environ.get("GPU_TYPE", "NVIDIA")

        def get_gpu(resource_gpu):
            try:
                if resource_gpu:
                    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA")  # TENCENT
                    if gpu_type == 'NVIDIA':
                        num = int(resource_gpu.split(',')[0])
                        num = 2 if num>2 else num
                        return num, num
                    if gpu_type == 'TENCENT':
                        core = int(resource_gpu.split(',')[0])
                        memory = int(resource_gpu.split(',')[1]) if ',' in resource_gpu else 0
                        return core, memory
            except Exception as e:
                print(e)

            return 0, 0


        resources_requests = {
            "cpu": requests_cpu,
            "memory": requests_memory
        }
        resources_limits = {
            "cpu": requests_cpu,
            "memory": requests_memory
        }

        if gpu_type == 'NVIDIA':
            gpu_num = get_gpu(resource_gpu)[0]
            if gpu_num:
                resources_requests['nvidia.com/gpu'] = str(gpu_num)
                resources_limits['nvidia.com/gpu'] = str(gpu_num)

        if gpu_type=='TENCENT':
            core = get_gpu(resource_gpu)[0]
            memory = 4*get_gpu(resource_gpu)[1]
            if core and memory:
                resources_requests['tencent.com/vcuda-core'] = str(core)
                resources_requests['tencent.com/vcuda-memory'] = str(memory)
                resources_limits['tencent.com/vcuda-core'] = str(core)
                resources_limits['tencent.com/vcuda-memory'] = str(memory)

        resources_tencent = client.V1ResourceRequirements(requests=resources_requests, limits=resources_limits)
        resources_obj=resources_tencent

        if ports:
            if type(ports)==str:
                ports = [int(port) for port in ports.split(',')]
            ports = [client.V1ContainerPort(name='port%s' % index, protocol='TCP', container_port=port) for index, port in
                 enumerate(ports)] if ports else None
        else:
            ports=None

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
            ports=ports
        )

        return container


    # @pysnooper.snoop()
    def make_pod(self,namespace,name,labels,command,args,volume_mount,working_dir,node_selector,resource_memory,resource_cpu,resource_gpu,image_pull_policy,image_pull_secrets,image,hostAliases,env,privileged,accounts,username,ports=None,restart_policy='OnFailure',scheduler_name='default-scheduler'):
        annotations = None
        if scheduler_name == 'kube-batch':
            annotations = {
                'scheduling.k8s.io/group-name': name
            }
        metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels, annotations=annotations)
        image_pull_secrets = [client.V1LocalObjectReference(image_pull_secret) for image_pull_secret in image_pull_secrets]
        nodeSelector = None
        if node_selector and '=' in node_selector:
            nodeSelector={}
            for selector in re.split(',|;|\n|\t', node_selector):
                nodeSelector[selector.strip().split('=')[0].strip()]=selector.strip().split('=')[1].strip()

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
                                          ports=ports)]

        # 添加host
        host_aliases = []
        if hostAliases:
            hostAliases_list = re.split('\r|\n', hostAliases)
            for row in hostAliases_list:
                hosts = row.split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    host_aliase = client.V1HostAlias(ip=hosts[0], hostnames=hosts[1:])
                    host_aliases.append(host_aliase)

        service_account = accounts if accounts else None
        spec = v1_pod_spec.V1PodSpec(image_pull_secrets=image_pull_secrets, node_selector=nodeSelector,
                                     volumes=k8s_volumes, containers=containers, restart_policy=restart_policy,
                                     host_aliases=host_aliases, service_account=service_account,scheduler_name=scheduler_name)
        pod = v1_pod.V1Pod(api_version='v1', kind='Pod', metadata=metadata, spec=spec)
        return pod,spec


    # @pysnooper.snoop()
    def create_debug_pod(self,namespace,name,labels,command,args,volume_mount,working_dir,node_selector,resource_memory,resource_cpu,resource_gpu,image_pull_policy,image_pull_secrets,image,hostAliases,env,privileged,accounts,username,scheduler_name='default-scheduler'):
        try:
            self.v1.delete_namespaced_pod(name, namespace=namespace,grace_period_seconds=0)
            time.sleep(1)
        except Exception as e:
            print(e)
        pod,pod_spec = self.make_pod(
            namespace=namespace,
            name =name,
            labels=labels,
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
            scheduler_name=scheduler_name
        )
        # print(pod)
        pod = self.v1.create_namespaced_pod(namespace,pod)
        time.sleep(1)

    # 创建hubsecret
    # @pysnooper.snoop()
    def apply_hubsecret(self,namespace,name,user,password,server):
        try:
            hubsecrest = self.v1.read_namespaced_secret(name=name,namespace=namespace)
            if hubsecrest:
                self.v1.delete_namespaced_secret(name,namespace=namespace)
        except Exception as e:
            print(e)

        cred_payload = {
            "auths": {
                server: {
                    "username": user,
                    "password": password,
                    "auth": base64.b64encode((user+":"+password).encode()).decode(),
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

        secret_objects = self.v1.create_namespaced_secret(namespace=namespace,body=secret)
        return secret_objects


    # 创建notebook
    def create_crd(self,group,version,plural,namespace,body):
        self.crd = client.CustomObjectsApi()
        crd_objects = self.crd.create_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural,body=body)
        return crd_objects

    # 创建pod
    def create_pod(self,namespace,body):
        pod_object = self.v1.create_namespaced_pod(namespace=namespace,body=body)
        return pod_object

    #
    # def get_deployment(self,name,namespace):
    #     client.AppsV1Api().  (name, namespace)
    #     return []

    # 删除deployment
    def delete_deployment(self,namespace,name=None,labels=None):
        if name:
            try:
                client.AppsV1Api().delete_namespaced_deployment(name=name, namespace=namespace)
            except Exception as e:
                print(e)
        elif labels:
            try:
                labels_arr = ["%s=%s"%(key,labels[key]) for key in labels]
                labels_str=','.join(labels_arr)
                deploys = self.AppsV1Api.list_namespaced_deployment(namespace=namespace,label_selector=labels_str).items
                for deploy in deploys:
                    client.AppsV1Api().delete_namespaced_deployment(name = deploy.metadata.name, namespace=namespace)
            except Exception as e:
                print(e)

    # @pysnooper.snoop(watch_explode=())
    def create_deployment(self,namespace,name,replicas,labels,command,args,volume_mount,working_dir,node_selector,resource_memory,resource_cpu,resource_gpu,image_pull_policy,image_pull_secrets,image,hostAliases,env,privileged,accounts,username,ports,scheduler_name='default-scheduler'):
        pod,pod_spec = self.make_pod(
            namespace=namespace,
            name=name,
            labels=labels,
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
            scheduler_name=scheduler_name
        )
        pod_spec.restart_policy='Always'  # dp里面必须是Always
        dp_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels)
        selector = client.models.V1LabelSelector(match_labels={"app":name,'user':username})
        template_metadata = v1_object_meta.V1ObjectMeta(labels={"app":name,'user':username})
        template = client.models.V1PodTemplateSpec(metadata=template_metadata,spec=pod_spec)
        dp_spec = apps_v1beta1_deployment_spec.AppsV1beta1DeploymentSpec(replicas=int(replicas), selector=selector,template=template)
        dp = apps_v1beta1_deployment.AppsV1beta1Deployment(api_version='apps/v1', kind='Deployment', metadata=dp_metadata, spec=dp_spec)
        # print(dp.to_str())
        try:
            client.AppsV1Api().delete_namespaced_deployment(name, namespace)
        except Exception as e:
            print(e)

        try:
            dp = client.AppsV1Api().create_namespaced_deployment(namespace, dp)
        except Exception as e:
            print(e)
            try:
                client.AppsV1Api().patch_namespaced_deployment(name=name,namespace=namespace,body=dp)
            except Exception as e1:
                print(e1)
        # time.sleep(2)


    # 删除statefulset
    # @pysnooper.snoop()
    def delete_statefulset(self,namespace,name=None,labels=None):
        if name:
            try:
                client.AppsV1Api().delete_namespaced_stateful_set(name=name, namespace=namespace)
            except Exception as e:
                print(e)
        elif labels:
            try:
                labels_arr = ["%s=%s"%(key,labels[key]) for key in labels]
                labels_str=','.join(labels_arr)
                stss = self.AppsV1Api.list_namespaced_stateful_set(namespace=namespace,label_selector=labels_str).items
                for sts in stss:
                    client.AppsV1Api().delete_namespaced_stateful_set(name = sts.metadata.name, namespace=namespace)
            except Exception as e:
                print(e)

    # @pysnooper.snoop(watch_explode=())
    def create_statefulset(self,namespace,name,replicas,labels,command,args,volume_mount,working_dir,node_selector,resource_memory,resource_cpu,resource_gpu,image_pull_policy,image_pull_secrets,image,hostAliases,env,privileged,accounts,username,ports,restart_policy='Always',scheduler_name='default-scheduler'):

        pod,pod_spec = self.make_pod(
            namespace=namespace,
            name=name,
            labels=labels,
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
            restart_policy=restart_policy,
            scheduler_name=scheduler_name
        )
        annotations = None
        if scheduler_name == 'kube-batch':
            annotations = {
                'scheduling.k8s.io/group-name': name
            }
        sts_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels=labels)
        selector = client.models.V1LabelSelector(match_labels={"app":name,'user':username})
        template_metadata = v1_object_meta.V1ObjectMeta(labels={"app":name,'user':username},annotations=annotations)
        template = client.models.V1PodTemplateSpec(metadata=template_metadata,spec=pod_spec)
        sts_spec = client.models.V1StatefulSetSpec(pod_management_policy='Parallel',replicas=int(replicas), selector=selector,template=template,service_name=name)
        sts = client.models.V1StatefulSet(api_version='apps/v1', kind='StatefulSet', metadata=sts_metadata, spec=sts_spec)
        # print(dp.to_str())
        try:
            client.AppsV1Api().delete_namespaced_stateful_set(name, namespace)
        except Exception as e:
            pass
            # print(e)

        try:
            sts = client.AppsV1Api().create_namespaced_stateful_set(namespace, sts)
        except Exception as e:
            print(e)



    # 创建pod
    # @pysnooper.snoop()
    def create_service(self,namespace,name,username,ports,selector=None):
        svc_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels={"app":name,'user':username})
        ports = [client.V1ServicePort(name='http%s'%index, port=int(port), protocol='TCP', target_port=int(port)) for index,port in enumerate(ports)]
        svc_spec = client.V1ServiceSpec(ports=ports, selector={"app": name, 'user': username}, type='ClusterIP')
        if selector:
            svc_spec = client.V1ServiceSpec(ports=ports, selector=selector, type='ClusterIP')
        service = client.V1Service(api_version='v1', kind='Service', metadata=svc_metadata, spec=svc_spec)
        # print(service.to_dict())
        try:
            self.v1.delete_namespaced_service(name, namespace)
        except Exception as e:
            print(e)
        try:
            service = self.v1.create_namespaced_service(namespace, service)
        except Exception as e:
            print(e)


    # @pysnooper.snoop()
    def create_headless_service(self,namespace,name,username,run_id):
        svc_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels={"app":name,'user':username,"run-id":run_id})
        svc_spec = client.V1ServiceSpec(cluster_ip='None', selector={"app":name,'user':username},type='ClusterIP')
        service = client.V1Service(api_version='v1', kind='Service', metadata=svc_metadata, spec=svc_spec)
        print(service.to_dict())
        try:
            self.v1.delete_namespaced_service(name, namespace)
        except Exception as e:
            pass
            # print(e)
        try:
            service = self.v1.create_namespaced_service(namespace, service)
        except Exception as e:
            print(e)


    # 创建pod
    # @pysnooper.snoop()
    def create_ingress(self,namespace,name,host,username,port):
        self.v1beta1 = client.ExtensionsV1beta1Api()
        ingress_metadata = v1_object_meta.V1ObjectMeta(name=name, namespace=namespace, labels={"app":name,'user':username},annotations={"nginx.ingress.kubernetes.io/proxy-connect-timeout":"3000","nginx.ingress.kubernetes.io/proxy-send-timeout":"3000","nginx.ingress.kubernetes.io/proxy-read-timeout":"3000","nginx.ingress.kubernetes.io/proxy-body-size":"1G"})
        backend = client.ExtensionsV1beta1IngressBackend(service_name=name,service_port=port)
        path = client.ExtensionsV1beta1HTTPIngressPath(backend=backend,path='/')
        http = client.ExtensionsV1beta1HTTPIngressRuleValue(paths=[path])
        rule = client.ExtensionsV1beta1IngressRule(host=host, http=http)
        ingress_spec = client.ExtensionsV1beta1IngressSpec(rules=[rule])
        ingress = client.ExtensionsV1beta1Ingress(api_version='extensions/v1beta1', kind='Ingress', metadata=ingress_metadata, spec=ingress_spec)
        print(ingress.to_dict())
        try:
            self.v1beta1.delete_namespaced_ingress(name = name,namespace= namespace)
        except Exception as e:
            print(e)

        try:
            ingress = self.v1beta1.create_namespaced_ingress(namespace = namespace, body=ingress)
        except Exception as e:
            print(e)


    #
    def delete_istio_ingress(self,namespace,name):
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
    def create_istio_ingress(self,namespace,name,host,ports):
        crd_info={
            "group": "networking.istio.io",
            "version": "v1alpha3",
            "plural": "virtualservices",
            'kind': 'VirtualService',
            "timeout": 60 * 60 * 24 * 1
        }

        crd_list = self.get_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                               namespace=namespace, return_dict=None)
        for vs_obj in crd_list:
            if vs_obj['name'] == name or vs_obj['name']== name+"-8080":
                self.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                               namespace=namespace, name=vs_obj['name'])
                time.sleep(1)

        if len(ports)>0:
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
                                        "host": "%s.%s.svc.cluster.local" % (name,namespace),
                                        "port": {
                                            "number": int(ports[0])
                                        }
                                    }
                                }
                            ],
                            "timeout": "300s"
                        }
                    ]
                }
            }
            crd = self.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                                 namespace=namespace, body=crd_json)
        if len(ports)>1:
            crd_json = {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "VirtualService",
                "metadata": {
                    "name": name+"-8080",
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
                            "timeout": "300s"
                        }
                    ]
                }
            }
            crd = self.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                                 namespace=namespace, body=crd_json)





    def to_memory_GB(self,memory):
        if 'Ki' in memory:
            return float(memory.replace('Ki',''))/1000000
        if 'Mi' in memory:
            return float(memory.replace('Mi',''))/1000
        if 'Gi' in memory:
            return float(memory.replace('Gi',''))
        return 0

    @pysnooper.snoop(watch_explode=('item'))
    def get_node_metrics(self):
        back_metrics=[]
        cust = client.CustomObjectsApi()
        metrics = cust.list_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'nodes')  # All node metrics
        items=metrics['items']
        for item in items:
            back_metrics.append({
                "name":item['metadata']['name'],
                "time":item['timestamp'],
                "cpu": int(item['usage']['cpu'].replace('n',''))/1000000,
                "memory": self.to_memory_GB(item['usage']['memory']),
                "window": item['window'],
            })
        # print(back_metrics)
        return back_metrics


    def get_pod_metrics(self,namespace=None):
        back_metrics = []
        cust = client.CustomObjectsApi()
        if namespace:
            metrics = cust.list_namespaced_custom_object('metrics.k8s.io', 'v1beta1', namespace,'pods')  # Just pod metrics for the default namespace
        else:
            metrics = cust.list_cluster_custom_object('metrics.k8s.io', 'v1beta1', 'pods')  # All Pod Metrics
        items=metrics.get('items',[])
        # print(items)
        for item in items:
            back_metrics.append({
                "name":item['metadata']['name'],
                "time":item['timestamp'],
                "namespace":item['metadata']['namespace'],
                "cpu": sum(int(container['usage']['cpu'].replace('n',''))/1000000 for container in item['containers']),
                "memory": sum(self.to_memory_GB(container['usage']['memory']) for container in item['containers']),
                "window": item['window']
            })
        # print(back_metrics)
        return back_metrics



    @pysnooper.snoop()
    def exec_command(self,name,namespace,command):
        try:
            resp = self.v1.read_namespaced_pod(name=name,namespace=namespace)
        except ApiException as e:
            if e.status != 404:
                print("Unknown error: %s" % e)
                return


        self.v1.connect_get_namespaced_pod_exec(
            name,
            namespace,
            command=command,
            # stderr = True,
            # stdin = True,
            # stdout = True,
            # tty = True
        )

    # 实时跟踪指定pod日志，直到pod结束
    def watch_pod_log(self,name,namespace,):
        print('begin follow log')
        w = watch.Watch()
        for event in w.stream(self.v1.read_namespaced_pod_log, name=name, namespace=namespace):
            print(event)

        print('end follow log')

    def watch_tfjob_log(self,name,namespace,):
        print('begin follow log')
        w = watch.Watch()
        for event in w.stream(self.v1.read_namespaced_pod_log, name=name, namespace=namespace):
            print(event)

        print('end follow log')

    def get_uesd_gpu(self,namespaces):

        all_gpu_pods = []

        def get_used_gpu(pod):
            name = pod.metadata.name
            user = pod.metadata.labels.get('run-rtx', '')
            if not user:
                user = pod.metadata.labels.get('user', '')
            if not user:
                user = pod.metadata.labels.get('rtx-user','')
            containers = pod.spec.containers

            gpu = 0
            for container in containers:
                limits = container.resources.limits
                request = container.resources.requests
                container_gpu = 0
                if limits:
                    container_gpu = int(limits.get('tencent.com/vcuda-core', 0)) / 100
                    if not container_gpu:
                        container_gpu = int(limits.get('nvidia.com/gpu', 0))
                elif request:
                    container_gpu = int(request.get('tencent.com/vcuda-core', 0)) / 100
                    if not container_gpu:
                        container_gpu = int(request.get('nvidia.com/gpu', 0))
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



# @pysnooper.snoop()
def check_status_time(status,hour=8):
    if type(status)==dict:
        for key in status:
            try:
                if key=='startedAt' or key=='finishedAt':
                    if type(status[key])==datetime.datetime:
                        status[key]=status[key]-datetime.timedelta(hours=hour)
                    elif type(status[key])==str:
                        status[key] = (datetime.datetime.strptime(status[key].replace('T',' ').replace('Z',''),'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(e)
            status[key] = check_status_time(status[key],hour)

    elif type(status)==list:
        for index in range(len(status)):
            status[index] = check_status_time(status[index],hour)

    return status

#
# if __name__=='__main__':
#     k8s_client = K8s(file_path='~/.kube/config')
#     k8s_client.get_node_metrics()



