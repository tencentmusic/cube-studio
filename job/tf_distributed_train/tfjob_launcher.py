# -*- coding: utf-8 -*-
import datetime
import json
import subprocess
import time
import uuid

from job.pkgs.constants import (DEF_IMAGE_PULL_SECRETS, JOB_DEF_NAMESPACE,
                                WORKER_DEF_RESOURCE_LIMITS, ComputeResource,
                                NodeAffnity, PodAffnity)
from job.pkgs.context import JobComponentRunner, KFJobContext
from job.pkgs.k8s.tfjob import TFJob
from job.pkgs.utils import parse_timedelta
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

TRAINER_TYPE_PLAIN = "plain"
TRAINER_TYPE_RUNNER = "runner"

TRAINER_SPECS = {
    TRAINER_TYPE_PLAIN: {
        "tlinux": {
            "image": "csighub.tencentyun.com/tme-kubeflow/tf_plain_train:20201010",
            "cmd": ["python3", "-m", "job.tf_plain_train.plain_train"]
        },
        "ubt": {
            "image": "csighub.tencentyun.com/tme-kubeflow/tf_plain_train_ubt:20201010",
            "cmd": ["python3", "-m", "job.tf_plain_train.plain_train"]
        }
    },
    TRAINER_TYPE_RUNNER: {
        "tlinux": {
            "image": "csighub.tencentyun.com/tme-kubeflow/tf_keras_train:20201010",
            "cmd": ["python3", "-m", "job.tf_keras_train.runner_train"]
        },
        "ubt": {
            "image": "csighub.tencentyun.com/tme-kubeflow/tf_keras_train_ubt:20201010",
            "cmd": ["python3", "-m", "job.tf_keras_train.runner_train"]
        }
    }
}


class TFJobLauncher(JobComponentRunner):
    def job_func(self, jc_entry):
        job = jc_entry.job
        job_name = job.get('name')
        job_namespace = job.get('namespace') or jc_entry.context.namespace or JOB_DEF_NAMESPACE
        num_workers = int(job.get('num_workers', 1))
        num_pss = int(job.get('num_pss', 0))
        node_affin = job.get("node_affin")
        pod_affin = job.get("pod_affin")
        node_selector = job.get("node_selector", {}) or jc_entry.context.parsed_node_selector()
        resources = job.get("resources")
        debug = job.get("debug", False)
        if not isinstance(resources, dict) or 'limits' not in resources:
            print("user specified resource {} not valid".format(resources))

            resources = jc_entry.context.parsed_resource_spec()
            if resources:
                print("will use resource spec from tfjob for workers: {}".format(resources))
            else:
                resources = WORKER_DEF_RESOURCE_LIMITS

        ps_resources = job.get("ps_resources")
        chief_resources = job.get("chief_resources")

        if (ComputeResource.P_GPU in resources['limits'] or ComputeResource.V_GPU_CORE in resources['limits']) \
                and not node_affin:
            node_affin = NodeAffnity.ONLY_GPU
            print("auto set node_affin={}".format(node_affin))

        if node_affin in [NodeAffnity.ONLY_GPU, NodeAffnity.PREF_GPU] and 'cpu' in node_selector:
            node_selector.pop('cpu', None)
            print("auto poped up 'cpu' in node selector: {}".format(node_selector))

        if node_affin in [NodeAffnity.ONLY_CPU, NodeAffnity.PREF_CPU] and 'gpu' in node_selector:
            node_selector.pop('gpu', None)
            print("auto poped up 'gpu' in node selector: {}".format(node_selector))

        restart_policy = job.get("restart_policy", '').strip()
        if restart_policy and restart_policy not in ['OnFailure', 'Always', 'ExitCode', 'Never']:
            print("WARNING: unrecognized 'restart_policy' '{}', reset to 'Never'".format(restart_policy))
            restart_policy = 'Never'

        backoff_limits = job.get("backoff_limits", num_workers)
        if backoff_limits < 0:
            print("WARNING: 'backoff_limits' should be >=0, got {}, defaults to 1".format(backoff_limits))
            backoff_limits = 1

        job_timeout = parse_timedelta(job.get('timeout', '365d'))
        job_polling_interval = parse_timedelta(job.get('polling_interval', '30s'))
        trainer_type = job.get("trainer", TRAINER_TYPE_RUNNER).strip().lower()
        os_ver = job.get("os_ver", "ubt").strip().lower()

        trainer_spec = TRAINER_SPECS.get(trainer_type)
        if not trainer_spec:
            raise NotImplementedError("unsupported trainer type '{}', supported are {}"
                                      .format(trainer_type, TRAINER_SPECS.keys()))

        trainer_spec = trainer_spec.get(os_ver)
        if not trainer_spec:
            raise NotImplementedError("unsupported os_ver '{}', supported are {}"
                                      .format(os_ver, trainer_spec.keys()))

        print("use trainer '{}', spec={}, num_workers={}, num_pss={}"
              .format(trainer_type, trainer_spec, num_workers, num_pss))

        driver_job_detail = job.get('job_detail')

        driver_args = [
            "--job", json.dumps(driver_job_detail),
            "--pack-path", jc_entry.pack_path,
            # "--upstream-output-file", jc_entry.upstream_output_file,
            "--export-path", jc_entry.export_path,
            "--pipeline-id", jc_entry.pipeline_id,
            "--run-id", jc_entry.run_id,
            "--creator", jc_entry.creator,
            # "--output-file", jc_entry.output_file or self.output_file
        ]

        driver_mounts = jc_entry.context.parsed_volumn_mounts() or []

        job_labels = {
            "run-rtx": jc_entry.runner,
            "upload-rtx": jc_entry.creator,
            "pipeline-id": jc_entry.pipeline_id,
            "run-id": jc_entry.run_id,
            "workflow-name": jc_entry.pipeline_name,
            'task-id': jc_entry.task_id,
            'task-name': jc_entry.task_name
        }

        user_envs = job.get("envs")
        driver_envs = jc_entry.context.to_k8s_env_list()
        if isinstance(user_envs, dict):
            for k, v in user_envs.items():
                driver_envs.append({"name": str(k), "value": str(v)})
                
        ps_config = job.get("ps_config", {})
        if ps_config:
            driver_envs.append({"name": "PS_CONFIG", "value": json.dumps(ps_config)})

        if 'profile_batch' in driver_job_detail.get('train_args', {}).get('tensorboard', {}) and \
                node_affin in [NodeAffnity.ONLY_GPU, NodeAffnity.PREF_GPU]:
            privileged = True
            print("job use gpu and tf profiler, set privileged=True")
        else:
            privileged = job.get('privileged', False)
            print("privileged={}".format(privileged))

        driver_image = trainer_spec.get("image")
        tf_ver = job.get('tf_ver', '').strip()
        if tf_ver:
            driver_image = '-'.join([driver_image, tf_ver])
            
        driver_cmd = trainer_spec.get("cmd")
        if debug and os_ver == 'ubt':
            driver_cmd = ["valgrind", "--tool=memcheck", "--leak-check=full", 
                          "--show-reachable=yes", "--trace-children=yes"] + driver_cmd
            print("turned on debug mode, driver_cmd={}".format(driver_cmd))

        self.launch_tfjob(job_name, job_namespace, num_workers, num_pss, driver_image,
                          driver_cmd, driver_args, driver_envs, driver_mounts, resources,
                          restart_policy, node_affin, pod_affin, job_labels, backoff_limits, job_timeout,
                          job_polling_interval, False, node_selector, privileged, jc_entry.creator,
                          ps_resources, chief_resources)

    @classmethod
    def default_job_name(cls):
        import re
        ctx = KFJobContext.get_context()
        p_name = str(ctx.pipeline_name) or ''
        p_name = re.sub(r'[^-a-z0-9]', '-', p_name)
        jid = str(uuid.uuid4()).replace('-', '')
        name = "-".join(["tfjob", p_name, jid])[:54]
        if name.endswith('-'):
            name = name[:-1]
        return name
        # return "tfjob-" + str(uuid.uuid1())

    @classmethod
    def launch_tfjob(cls, name, namespace, num_workers, num_pss, driver_image, driver_cmd,
                     driver_args, driver_envs, driver_mounts, resources=None, restart_policy=None,
                     node_affin=None, pod_affin=None, job_labels={}, backoff_limits=3, job_timeout=None,
                     job_polling_interval=None, delete_after_finish=False, node_selector={}, privileged=False,
                     creator='', ps_resources=None, chief_resources=None):

        subprocess.check_call("echo '10.101.140.98 cls-g9v4gmm0.ccs.tencent-cloud.com' >> /etc/hosts", shell=True)

        k8s_config.load_incluster_config()
        k8s_api_client = k8s_client.ApiClient()
        tfjob = TFJob("v1", k8s_api_client)
        job_name = name.strip() if name and name.strip() else cls.default_job_name()

        if node_affin == NodeAffnity.PREF_GPU:
            node_affin = NodeAffnity.ONLY_GPU
            print("WARING: 'node_affin' set to 'pref_gpu', changed it to 'only_gpu' to avoid heterogeneity")
        if node_affin == NodeAffnity.PREF_CPU:
            node_affin = NodeAffnity.ONLY_CPU
            print("WARING: 'node_affin' set to 'pref_cpu', changed it to 'only_cpu' to avoid heterogeneity")

        if not pod_affin and node_affin in [NodeAffnity.ONLY_GPU, NodeAffnity.PREF_GPU]:
            pod_affin = PodAffnity.CONCENT
            print("auto set pod_affin to {}".format(pod_affin))

        print("begin trying to delete old tfjobs of run-id='{}', task-id='{}'"
              .format(job_labels['run-id'], job_labels['task-id']))
        tfjob.delete(labels={'run-id': job_labels['run-id'], 'task-id': job_labels['task-id']})
        print("trying to delete old tfjobs of run-id='{}', task-id='{}' finished"
              .format(job_labels['run-id'], job_labels['task-id']))

        st = time.perf_counter()
        print('begin create new tfjob %s' % job_name)
        tfjob.create(job_name, namespace, num_workers, num_pss, driver_image, driver_cmd, driver_args,
                     driver_envs, driver_mounts, resources, restart_policy, DEF_IMAGE_PULL_SECRETS,
                     node_affin, pod_affin, job_labels, backoff_limits, node_selector, privileged, creator,
                     ps_resources, chief_resources)

        job_timeout = job_timeout if job_timeout else datetime.timedelta(days=365)
        job_polling_inteval = job_polling_interval if job_polling_interval else datetime.timedelta(seconds=30)

        condition = tfjob.wait_for_condition(namespace, job_name, ["Succeeded", "Failed"], job_timeout,
                                             job_polling_inteval, trace_worker_log=True)

        print("TFJob '{}' finished in condition '{}', cost {}s".format(job_name, condition, time.perf_counter() - st))

        if condition != 'Succeeded':
            raise RuntimeError("TFJob '{}' in namespace '{}' failed, num_workers={}, driver_args={}"
                               .format(job_name, namespace, num_workers, driver_args))
        if delete_after_finish:
            print("will delete tfjob '{}' in '{}'".format(job_name, namespace))
            tfjob.delete(name=job_name, namespace=namespace)
            print("deleted tfjob '{}' in '{}'".format(job_name, namespace))


if __name__ == "__main__":
    runner = TFJobLauncher("TFJob launcher for train component")
    runner.run()
