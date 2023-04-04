# coding=utf-8
# @Time     : 2020/12/28 21:05
# @Auther   : lionpeng@tencent.com

from .k8s_crd import K8sCRD
from ..constants import NodeAffnity
import os

class PyTorchJob(K8sCRD):
    def __init__(self, version, k8s_api_client):
        super(PyTorchJob, self).__init__("kubeflow.org", "pytorchjobs", version, k8s_api_client)

    def create(self, name, namespace, num_workers, driver_image, driver_command, driver_args, driver_envs,
               driver_mounts, resources, restart_policy, image_pull_secrets=None, node_affin=None,
               pod_affin=None, labels={}, backoff_limits=3, node_selector={}, creator=''):

        if node_affin in [NodeAffnity.ONLY_GPU, NodeAffnity.PREF_GPU]:
            resources = resources or {}
            if "limits" not in resources or "nvidia.com/gpu" not in resources['limits']:
                if "limits" not in resources:
                    resources['limits'] = {"nvidia.com/gpu": 1}
                else:
                    resources['limits'].update({"nvidia.com/gpu": 1})
                print("injected gpu resource spec: {}".format(resources))
        cr_spec = {
            "apiVersion": "{}/{}".format(self.group, self.version),
            "kind": "PyTorchJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": labels or {}
            },
            "spec": {
                "cleanPodPolicy": "None",
                "backoffLimit": backoff_limits,
                "pytorchReplicaSpecs": {}
            }
        }

        cr_spec["spec"]["pytorchReplicaSpecs"]["Master"] = self.make_worker_spec(
            name, 1, restart_policy, node_affin, pod_affin, driver_image,
            driver_command, driver_args, driver_envs, driver_mounts, resources,
            image_pull_secrets, node_selector)
        if num_workers > 0:
            print("specified {} worker nodes".format(num_workers))
            cr_spec["spec"]["pytorchReplicaSpecs"]["Worker"] = self.make_worker_spec(
                name, num_workers, restart_policy, node_affin, pod_affin, driver_image,
                driver_command, driver_args, driver_envs, driver_mounts, resources,
                image_pull_secrets, node_selector)
        else:
            print("specified no worker nodes")

        return super(PyTorchJob, self).create(cr_spec)

    def make_worker_spec(self, job_name, replicas, restart_policy, node_affin, pod_affin, driver_image,
                         driver_command, driver_args, driver_envs, driver_mounts, resources=None,
                         image_pull_secrets=None, node_selector={}, creator=''):
        worker_spec = {
            "replicas": replicas,
            "restartPolicy": restart_policy or 'Never',
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    },
                    "labels": {
                        "app": job_name,
                        "run-id": os.getenv('KFJ_RUN_ID', 'unknown'),
                        'rtx-user': os.getenv('KFJ_RUNNER','unknown'),
                        "pipeline-id": os.getenv('KFJ_PIPELINE_ID', 'unknown'),
                        "pipeline-name": os.getenv('KFJ_PIPELINE_NAME', 'unknown'),
                        "task-id": os.getenv('KFJ_TASK_ID', 'unknown'),
                        "task-name": os.getenv('KFJ_TASK_NAME', 'unknown')
                    }
                },
                "spec": {
                    "schedulerName": "kube-batch",
                    "affinity": self.make_affinity_spec(job_name, node_affin, pod_affin),
                    "containers": [
                        {
                            "name": "pytorch",
                            "image": driver_image,
                            "imagePullPolicy": "Always",
                            "command": driver_command,
                            "args": driver_args,
                            "env": driver_envs or [],
                            "volumeMounts": [
                                self.make_volume_mount_spec(mn, mt, mp, creator)[1] for mn, mt, mp in driver_mounts
                            ],
                            "resources": resources if resources else {}
                        }
                    ],
                    "volumes": [
                        self.make_volume_mount_spec(mn, mt, mp, creator)[0] for mn, mt, mp in driver_mounts
                    ],
                    "nodeSelector": node_selector or {}
                }
            }
        }
        if image_pull_secrets:
            worker_spec['template']['spec']['imagePullSecrets'] = [{'name': sec} for sec in image_pull_secrets]
        return worker_spec
