
import datetime
import time
import traceback
import re
import os
import json
import subprocess
from kubernetes import client as k8s_client

from ..exceptions.k8s_expections import K8SFailedException, K8SJOBTimeoutException
from ..constants import NodeAffnity, PodAffnity


class K8sCRD(object):
    """
    k8s custom resource抽象类
    """

    def __init__(self, group, plural, version, client):
        """

        Args:
            group: custom resource的group，对kubeflow来说，一般是kubeflow.org
            plural: custom resource类型的复数形式，例如tfjobs
            version: custom resource版本
            client: k8s客户端实例
        """
        self.group = group
        self.plural = plural
        self.version = version
        self.client = k8s_client.CustomObjectsApi(client)

    def _start_trace_worker_logs(self, namespace, name):
        try:
            subprocess.check_call("which stern", shell=True)
        except Exception as e:
            print("WARNING: found no stern installed, can not trace worker log: {}".format(e), flush=True)
            return None

        cmd = "stern {} --namespace {}"\
            .format(name, namespace)

        subproc = subprocess.Popen(cmd, universal_newlines=False, shell=True, bufsize=0)
        print("worker log tracer started: {}".format(subproc.pid), flush=True)
        return subproc

    def _keep_worker_log_tracer(self, namespace, name, trace_proc=None):
        if trace_proc is None:
            return None
        proc_ret = trace_proc.poll()
        if proc_ret is not None:
            print("worker log tracer {} is terminated with code {}, will restart it"
                  .format(trace_proc.pid, proc_ret), flush=True)
            print("*"*50, flush=True)
            return self._start_trace_worker_logs(namespace, name)
        return trace_proc

    def _stop_trace_worker_logs(self, trace_proc):
        if trace_proc is None:
            return
        proc_ret = trace_proc.poll()
        if proc_ret is not None:
            return
        trace_proc.kill()
        print("killed worker log tracer: {}".format(trace_proc.pid), flush=True)

    def wait_for_condition(self,
                           namespace,
                           name,
                           expected_conditions=[],
                           timeout=datetime.timedelta(days=365),
                           polling_interval=datetime.timedelta(seconds=30),
                           status_callback=None,
                           trace_worker_log=False):
        """Waits until any of the specified conditions occur.
        Args:
            namespace: namespace for the CR.
            name: Name of the CR.
            expected_conditions: A list of conditions. Function waits until any of the
                supplied conditions is reached.
            timeout: How long to wait for the CR.
            polling_interval: How often to poll for the status of the CR.
            status_callback: (Optional): Callable. If supplied this callable is
                invoked after we poll the CR. Callable takes a single argument which
                is the CR.
            trace_worker_log:
        """
        end_time = datetime.datetime.now() + timeout
        max_retries = 15
        retries = 0
        trace_proc = self._start_trace_worker_logs(namespace, name) if trace_worker_log else None
        trace_st = time.time()
        try:
            while True:
                try:
                    results = self.client.get_namespaced_custom_object(self.group, self.version, namespace,
                                                                       self.plural, name)
                except Exception as e:
                    if retries >= max_retries:
                        raise K8SFailedException("get k8s resource '{}/{}/{}' '{}' in namespace '{}' error: {}"
                                                 .format(self.group, self.version, self.plural, name, namespace, e))
                    print("get k8s resource '{}/{}/{}' '{}' in namespace '{}' error, will retry after 10s({}/{}):"
                          " {}\n{}".format(self.group, self.version, self.plural, name, namespace, retries,
                                           max_retries, e, traceback.format_exc()), flush=True)
                    retries += 1
                    time.sleep(10)
                    continue

                if results:
                    if status_callback:
                        status_callback(results)
                    expected, condition = self.is_expected_conditions(results, expected_conditions)
                    if expected:
                        print("k8s resource '{}/{}/{}' '{}' in namespace '{}' has reached the expected condition: '{}'"
                              .format(self.group, self.version, self.plural, name, namespace, condition),
                              flush=True)
                        return condition
                    else:
                        if trace_proc is None:
                            print("waiting k8s resource '{}/{}/{}' '{}' in namespace '{}' to reach conditions '{}',"
                                  " current is '{}'".format(self.group, self.version, self.plural, name, namespace,
                                                            expected_conditions, condition if condition else None),
                                  flush=True)
                    retries = 0
                elif retries < max_retries:
                    print("get k8s resource '{}/{}/{}' '{}' in namespace '{}' return empty, will retry after 10s({}/{})"
                          .format(self.group, self.version, self.plural, name, namespace, retries, max_retries),
                          flush=True)
                    retries += 1
                    continue
                else:
                    raise K8SFailedException("get k8s resource '{}/{}/{}' '{}' in namespace '{}' return empty"
                                             .format(self.group, self.version, self.plural, name, namespace))

                if datetime.datetime.now() + polling_interval > end_time:
                    raise K8SJOBTimeoutException("wating k8s resource '{}/{}/{}' '{}' in namespace '{}' to reach"
                                                 " conditions '{}' timeout, timeout={}, polling_interval={}"
                                                 .format(self.group, self.version, self.plural, name, namespace,
                                                         expected_conditions, timeout, polling_interval))
                time.sleep(polling_interval.total_seconds())
                trace_proc = self._keep_worker_log_tracer(namespace, name, trace_proc)
                if trace_proc is not None and time.time() - trace_st >= 3600*2:
                    print("will restart worker logger tracer", flush=True)
                    trace_proc.kill()
                    trace_proc = self._start_trace_worker_logs(namespace, name)
                    trace_st = time.time()
        finally:
            self._stop_trace_worker_logs(trace_proc)

    def is_expected_conditions(self, cr_object, expected_conditions):
        """
        判断cr是否达到指定状态，子类必须实现此类
        Args:
            cr_object: cr的json描述，通过api获得
            expected_conditions: 期望的状态列表

        Returns:
            tuple: is_expected, condition
        """
        conditions = cr_object.get('status', {}).get("conditions")
        if not conditions:
            return False, ""
        if conditions[-1]["type"] in expected_conditions and conditions[-1]["status"] == "True":
            return True, conditions[-1]["type"]
        else:
            return False, conditions[-1]["type"]

    def create(self, spec):
        """Create a CR.
        Args:
          spec: The spec for the CR.
        """
        try:
            # Create a Resource
            namespace = spec["metadata"].get("namespace", "default")
            name = spec["metadata"]["name"]
            print("creating k8s resource '{}/{}/{}' '{}' in namespace '{}'"
                  .format(self.group, self.version, self.plural, name, namespace))
            api_response = self.client.create_namespaced_custom_object(self.group, self.version, namespace,
                                                                       self.plural, spec)
            print("created k8s resource '{}/{}/{}' '{}' in namespace '{}'\nspec='{}'\nresponse={}"
                  .format(self.group, self.version, self.plural, name, namespace, spec, api_response))
            return api_response
        except Exception as e:
            print("create k8s resource '{}/{}/{}' error, spec={}: {}\n{}".format(self.group, self.version, self.plural,
                                                                                 spec, e, traceback.format_exc()))
            raise K8SFailedException("create k8s resource '{}/{}/{}' error, spec={}: {}"
                                     .format(self.group, self.version, self.plural, spec, e))

    def get_crd_status(self, crd_object, plural):
        status = ''
        # workflows 使用最后一个node的状态为真是状态
        if plural == 'workflows':
            if 'status' in crd_object and 'nodes' in crd_object['status']:
                keys = list(crd_object['status']['nodes'].keys())
                status = crd_object['status']['nodes'][keys[-1]]['phase']
                if status != 'Pending':
                    status = crd_object['status']['phase']
        elif plural == 'notebooks':
            if 'status' in crd_object and 'conditions' in crd_object['status'] and len(
                    crd_object['status']['conditions']) > 0:
                status = crd_object['status']['conditions'][0]['type']
        elif plural == 'inferenceservices':
            status = 'unready'
            if 'status' in crd_object and 'conditions' in crd_object['status'] and len(
                    crd_object['status']['conditions']) > 0:
                for condition in crd_object['status']['conditions']:
                    if condition['type'] == 'Ready' and condition['status'] == 'True':
                        status = 'ready'
        else:
            if 'status' in crd_object and 'phase' in crd_object['status']:
                status = crd_object['status']['phase']
            elif 'status' in crd_object and 'conditions' in crd_object['status'] and len(
                    crd_object['status']['conditions']) > 0:
                status = crd_object['status']['conditions'][-1]['type']  # tfjob和experiment是这种结构
        return status

    # @pysnooper.snoop(watch_explode=('crd_object'))
    def get_one_crd(self, namespace, name):
        self.crd = k8s_client.CustomObjectsApi()
        crd_object = self.crd.get_namespaced_custom_object(group=self.group, version=self.version, namespace=namespace,
                                                           plural=self.plural, name=name)

        # print(crd_object['status']['conditions'][-1]['type'])
        status = self.get_crd_status(crd_object, self.plural)

        creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
        creat_time = (
                    datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=8)).strftime(
            '%Y-%m-%d %H:%M:%S')

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

    def get_crd(self, namespace):
        self.crd = k8s_client.CustomObjectsApi()
        crd_objects = \
        self.crd.list_namespaced_custom_object(group=self.group, version=self.version, namespace=namespace,
                                               plural=self.plural)['items']
        back_objects = []
        for crd_object in crd_objects:
            # print(crd_object['status']['conditions'][-1]['type'])
            status = self.get_crd_status(crd_object, self.plural)

            creat_time = crd_object['metadata']['creationTimestamp'].replace('T', ' ').replace('Z', '')
            creat_time = (datetime.datetime.strptime(creat_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                hours=8)).strftime('%Y-%m-%d %H:%M:%S')

            back_object = {
                "name": crd_object['metadata']['name'],
                "namespace": crd_object['metadata']['namespace'] if 'namespace' in crd_object['metadata'] else '',
                "annotations": json.dumps(crd_object['metadata']['annotations'], indent=4,
                                          ensure_ascii=False) if 'annotations' in crd_object['metadata'] else '',
                "labels": json.dumps(crd_object['metadata']['labels'], indent=4, ensure_ascii=False) if 'labels' in
                                                                                                        crd_object[
                                                                                                            'metadata'] else '{}',
                "spec": json.dumps(crd_object['spec'], indent=4, ensure_ascii=False),
                "create_time": creat_time,
                "status": status,
                "status_more": json.dumps(crd_object['status'], indent=4,
                                          ensure_ascii=False) if 'status' in crd_object else ''
            }
            back_objects.append(back_object)
        return back_objects

    def delete(self, name=None, namespace='pipeline', labels=None):
        """
        delete a k8s cr
        Args:
            name: name of cr to be deleted
            namespace: namespace in which cr to be delete

        Returns:

        """
        if name:
            try:
                from kubernetes.client import V1DeleteOptions
                body = V1DeleteOptions(api_version=self.version, propagation_policy="Foreground")
                print("deleteing k8s resource '{}/{}/{}' '{}' in namespace '{}'"
                      .format(self.group, self.version, self.plural, name, namespace))
                api_response = self.client.delete_namespaced_custom_object(
                    self.group,
                    self.version,
                    namespace,
                    self.plural,
                    name,
                    body)
                print("deleted k8s resource '{}/{}/{}' '{}' in namespace '{}', response={}"
                      .format(self.group, self.version, self.plural, name, namespace, api_response))
                return api_response
            except Exception as e:
                print("delete k8s resource '{}/{}/{}' '{}' in namespace '{}' error: {}\n{}"
                      .format(self.group, self.version, self.plural, name, namespace, e, traceback.format_exc()))
                raise K8SFailedException("delete k8s resource '{}/{}/{}' '{}' in namespace '{}' error: {}"
                                         .format(self.group, self.version, self.plural, name, namespace, e))
        elif labels and type(labels) == dict:
            crds = self.get_crd(namespace=namespace)
            for crd in crds:
                if crd['labels']:
                    crd_labels = json.loads(crd['labels'])
                    for key in labels:
                        if key in crd_labels and labels[key] == crd_labels[key]:
                            try:
                                self.crd = k8s_client.CustomObjectsApi()
                                delete_body = k8s_client.V1DeleteOptions()
                                self.crd.delete_namespaced_custom_object(group=self.group, version=self.version,
                                                                         namespace=namespace, plural=self.plural,
                                                                         name=crd['name'], body=delete_body)
                            except Exception as e:
                                print(e)

    @staticmethod
    def make_affinity_spec(job_name, node_affin=None, pod_affin=None):
        affinity = {}
        if node_affin and node_affin.strip():
            node_affin = node_affin.strip().lower()
            if node_affin in [NodeAffnity.PREF_CPU, NodeAffnity.PREF_GPU]:
                affinity["nodeAffinity"] = {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "weight": 100,
                            "preference": {
                                "matchExpressions": [
                                    {
                                        "key": "cpu" if node_affin == NodeAffnity.PREF_CPU else "gpu",
                                        "operator": "In",
                                        "values": ["true"]
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif node_affin in [NodeAffnity.ONLY_CPU, NodeAffnity.ONLY_GPU]:
                affinity["nodeAffinity"] = {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "cpu" if node_affin == NodeAffnity.ONLY_CPU else "gpu",
                                        "operator": "In",
                                        "values": ["true"]
                                    }
                                ]
                            }
                        ]
                    }
                }

        if pod_affin and pod_affin.strip():
            pod_affin = pod_affin.strip().lower()
            if pod_affin == PodAffnity.SPREAD:
                affinity["podAntiAffinity"] = {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "weight": 100,
                            "podAffinityTerm": {
                                "labelSelector": {
                                    "matchExpressions": [
                                        {
                                            "key": "app",
                                            "operator": "In",
                                            "values": [job_name]
                                        }
                                    ]
                                },
                                "topologyKey": "kubernetes.io/hostname"
                            }
                        }
                    ]
                }
            elif pod_affin == PodAffnity.CONCENT:
                affinity["podAffinity"] = {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "weight": 100,
                            "podAffinityTerm": {
                                "labelSelector": {
                                    "matchExpressions": [
                                        {
                                            "key": "app",
                                            "operator": "In",
                                            "values": [job_name]
                                        }
                                    ]
                                },
                                "topologyKey": "kubernetes.io/hostname"
                            }
                        }
                    ]
                }

        return affinity

    @staticmethod
    def make_volume_mount_spec(mount_name, mount_type, mount_point, username):
        mount_type = mount_type.lower()
        if mount_type == 'pvc':
            vol_name = mount_name.replace('_', '-').lower()
            volume_spec = {
                "name": vol_name,
                "persistentVolumeClaim": {
                    "claimName": mount_name
                }
            }
            mount_spec = {
                "name": vol_name,
                "mountPath": os.path.join(mount_point, username),
                "subPath": username
            }
        elif mount_type == 'hostpath':
            temps = re.split(r'_|\.|/', mount_name)
            temps = [temp for temp in temps if temp]
            vol_name = '-'.join(temps).lower()
            volume_spec = {
                "name": vol_name,
                "hostPath": {
                    "path": mount_name
                }
            }
            mount_spec = {
                "name": vol_name,
                "mountPath": mount_point
            }
        elif mount_type == 'configmap':
            vol_name = mount_name.replace('_', '-').replace('/', '-').replace('.', '-').lower()
            volume_spec = {
                "name": vol_name,
                "configMap": {
                    "name": mount_name
                }
            }
            mount_spec = {
                "name": vol_name,
                "mountPath": mount_point
            }
        elif mount_type == 'memory':
            memory_size = mount_name.replace('_', '-').replace('/', '-').replace('.', '-').lower().replace('g', '')
            volumn_name = 'memory-%s' % memory_size
            volume_spec = {
                "name": volumn_name,
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": "%sGi" % memory_size
                }
            }

            mount_spec = {
                "name": volumn_name,
                "mountPath": mount_point
            }
        else:
            raise RuntimeError("unknown mount type '{}', only pvc/hostpath/configmap are allowed".format(mount_type))
        return volume_spec, mount_spec
