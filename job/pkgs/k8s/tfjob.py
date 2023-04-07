# -*- coding: utf-8 -*-

import copy

from job.pkgs.context import KFJobContext
from job.pkgs.utils import parse_size

from ..constants import ComputeResource, NodeAffnity, PodAffnity
from .k8s_crd import K8sCRD


class TFJob(K8sCRD):
    def __init__(self, version, k8s_api_client):
        super(TFJob, self).__init__("kubeflow.org", "tfjobs", version, k8s_api_client)

    def create(self, name, namespace, num_workers, num_pss, driver_image, driver_command, driver_args,
               driver_envs, driver_mounts, resources, restart_policy, image_pull_secrets=None, node_affin=None,
               pod_affin=None, labels={}, backoff_limits=3, node_selector={}, privileged=False, creator='',
               ps_resources=None, chief_resources=None):

        if node_affin in [NodeAffnity.ONLY_GPU, NodeAffnity.PREF_GPU]:
            ctx = KFJobContext.get_context()
            gup_type = (ctx.gpu_type or '').strip().upper()
            resources = resources or {}
            limits = resources.get('limits', {})
            if gup_type == ComputeResource.ClusterEnv.TKE:
                print("under tke environment")
                if ComputeResource.P_GPU in limits:
                    limits.pop(ComputeResource.V_GPU_CORE, None)
                    limits.pop(ComputeResource.V_GPU_MEM, None)
                    print("specified physical gpu, ignore v-gpu settings")
                else:
                    if ComputeResource.V_GPU_CORE not in limits:
                        limits[ComputeResource.V_GPU_CORE] = 100
                        print("{} not set, default to 100".format(ComputeResource.V_GPU_CORE))
                    else:
                        limits[ComputeResource.V_GPU_CORE] = int(limits[ComputeResource.V_GPU_CORE])
                    gpu_mem = parse_size(limits.get(ComputeResource.V_GPU_MEM, 0))
                    if not gpu_mem:
                        min_gpu_mem = parse_size(ctx.gpu_mem_min)
                        if not min_gpu_mem:
                            print("WARNING: {} not set and KFJ_GPU_MEM_MAX env are not set"
                                  .format(ComputeResource.V_GPU_MEM))
                            limits.pop(ComputeResource.V_GPU_MEM, None)
                        else:
                            gpu_mem = int((limits[ComputeResource.V_GPU_CORE] / 100 * min_gpu_mem) //
                                          ComputeResource.V_GPU_MEM_UNIT)
                            limits[ComputeResource.V_GPU_MEM] = gpu_mem
                            print("{} not set, set to {}".format(ComputeResource.V_GPU_MEM, gpu_mem))
                    else:
                        gpu_mem = int(gpu_mem // ComputeResource.V_GPU_MEM_UNIT)
                        limits[ComputeResource.V_GPU_MEM] = gpu_mem
            else:
                print("under idc environment")
                v_cores = limits.pop(ComputeResource.V_GPU_CORE, 100)
                limits.pop(ComputeResource.V_GPU_MEM, None)
                if ComputeResource.P_GPU not in limits:
                    limits[ComputeResource.P_GPU] = v_cores // 100
                    print("{} not set, default to 1".format(ComputeResource.P_GPU))
            resources['limits'] = limits
        else:
            resources = resources or {}
            limits = resources.get('limits', {})
            if limits:
                limits.pop(ComputeResource.P_GPU, None)
                limits.pop(ComputeResource.V_GPU_CORE, None)
                limits.pop(ComputeResource.V_GPU_MEM, None)
            print("cpu job, ignored gpu settings")

        print("resource spec: {}".format(resources))

        cr_spec = {
            "apiVersion": "{}/{}".format(self.group, self.version),
            "kind": "TFJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": labels or {}
            },
            "spec": {
                "cleanPodPolicy": "None",
                "backoffLimit": backoff_limits,
                "tfReplicaSpecs": {}
            }
        }

        worker_labels = labels or {}
        worker_labels['app'] = name

        if num_workers > 0:
            print("specified {} worker nodes".format(num_workers))
            cr_spec["spec"]["tfReplicaSpecs"]["Worker"] = self.make_worker_spec(
                name, num_workers, restart_policy, node_affin, pod_affin, driver_image,
                driver_command, driver_args, driver_envs, driver_mounts, resources,
                image_pull_secrets, node_selector, worker_labels, privileged, creator)
        else:
            raise RuntimeError("'num_workers' must be > 0, got {}".format(num_workers))

        if num_pss > 0:
            print("specified {} PS nodes".format(num_pss))
            if not ps_resources:
                ps_resources = copy.deepcopy(resources)
                print("ps node resources not specified, use worker resources as reference")
            limits = ps_resources.get('limits')
            if limits:
                limits.pop(ComputeResource.P_GPU, None)
                limits.pop(ComputeResource.V_GPU_CORE, None)
                limits.pop(ComputeResource.V_GPU_MEM, None)
            print("ps node resources: {}".format(ps_resources))

            cr_spec["spec"]["tfReplicaSpecs"]["PS"] = self.make_worker_spec(
                name, num_pss, restart_policy, NodeAffnity.ONLY_CPU, PodAffnity.SPREAD, 
                driver_image, driver_command, driver_args, driver_envs, driver_mounts, 
                ps_resources, image_pull_secrets, node_selector, worker_labels, privileged,
                creator)

            print("auto add chief node under ps training mode")
            if not chief_resources:
                chief_resources = copy.deepcopy(resources)
                print("chief node resources not specified, use worker resources as reference")
            limits = chief_resources.get('limits')
            if limits:
                limits.pop(ComputeResource.P_GPU, None)
                limits.pop(ComputeResource.V_GPU_CORE, None)
                limits.pop(ComputeResource.V_GPU_MEM, None)
            print("chief node resources: {}".format(chief_resources))

            cr_spec["spec"]["tfReplicaSpecs"]["Chief"] = self.make_worker_spec(
                name, 1, restart_policy, NodeAffnity.ONLY_CPU, pod_affin, driver_image,
                driver_command, driver_args, driver_envs, driver_mounts, chief_resources,
                image_pull_secrets, node_selector, worker_labels, privileged, creator)

        return super(TFJob, self).create(cr_spec)

    def make_worker_spec(self, job_name, replicas, restart_policy, node_affin, pod_affin, driver_image,
                         driver_command, driver_args, driver_envs, driver_mounts, resources=None,
                         image_pull_secrets=None, node_selector={}, labels={}, privileged=False, creator=''):
        worker_spec = {
            "replicas": replicas,
            "restartPolicy": restart_policy or 'Never',
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    },
                    "labels": labels or {}
                },
                "spec": {
                    "affinity": self.make_affinity_spec(job_name, node_affin, pod_affin),
                    "containers": [
                        {
                            "name": "tensorflow",
                            "image": driver_image,
                            "imagePullPolicy": "Always",
                            "command": driver_command,
                            "args": driver_args,
                            "env": driver_envs or [],
                            "volumeMounts": [
                                self.make_volume_mount_spec(mn, mt, mp, creator)[1] for mn, mt, mp in driver_mounts
                            ],
                            "resources": resources if resources else {},
                            "securityContext": {"allowPrivilegeEscalation": privileged}
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
