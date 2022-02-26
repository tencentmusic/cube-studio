
import os
import platform
import json
import sys
import codecs
import traceback
import re
import random
from .constants import ComputeResource
from .utils import parse_best_parameter

# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


class KFJobContext(object):
    __KFJ_ENVS = ['pipeline_id', 'run_id', 'creator', 'runner', 'model_repo_api_url',
                  'archive_base_path', 'pipeline_name', 'namespace', 'gpu_type', 'gpu_mem_max', 'gpu_mem_min',
                  'task_id', 'task_name', 'task_volume_mount', 'task_resource_cpu', 'task_resource_memory',
                  'task_resource_gpu', 'task_node_selector']

    __PIPELINE_ENVS = ['pack_path', 'export_path']

    __slots__ = __PIPELINE_ENVS + __KFJ_ENVS

    def __str__(self):
        d = {a: getattr(self, a) for a in self.__slots__}
        return str(d)

    def to_k8s_env_list(self):
        envs = []
        for n in self.__PIPELINE_ENVS:
            env_name = "PIPELINE_" + n.upper()
            if hasattr(self, n):
                envs.append({"name": env_name, "value": getattr(self, n)})

        for n in self.__KFJ_ENVS:
            env_name = "KFJ_" + n.upper()
            if hasattr(self, n):
                envs.append({"name": env_name, "value": getattr(self, n)})

        return envs

    def parsed_volumn_mounts(self):
        mounts_str = (self.task_volume_mount or '').strip()
        if not mounts_str:
            return None
        mounts = []
        for ms in re.split(r',|;', mounts_str):
            ms = ms.strip()
            if not ms:
                continue
            m = re.match(r"^([^\s]+)\s*\(\s*(pvc|hostpath|configmap|memory)\s*\)\s*:\s*([^\s]+)$", ms, re.I)
            if not m:
                continue
            mount_name = m.group(1)
            mount_type = m.group(2).lower()
            mount_point = m.group(3)
            mounts.append((mount_name, mount_type, mount_point))
        return mounts

    def parsed_resource_spec(self) -> dict:
        cpu = self.task_resource_cpu
        gpu = self.task_resource_gpu
        memory = self.task_resource_memory
        spec = {
            "limits": {
                "memory": memory,
                "cpu": cpu
            }
        }
        if gpu:
            fields = gpu.split(',')
            if len(fields) != 2:
                print("invalid gpu resource value '{}'".format(gpu))
            else:
                gpu_core = fields[0].strip()
                gpu_mem = fields[1].strip()
                try:
                    gpu_type = (self.context.gpu_type or '').strip().upper()
                    if gpu_type != ComputeResource.ClusterEnv.TKE:
                        spec['limits'][ComputeResource.P_GPU] = 1
                    else:
                        gpu_core = int(gpu_core)
                        gpu_mem = int(gpu_mem) * ComputeResource.V_GPU_MEM_UNIT
                        spec['limits'][ComputeResource.V_GPU_CORE] = gpu_core
                        spec['limits'][ComputeResource.V_GPU_MEM] = gpu_mem
                except Exception as e:
                    print("WARNING: parse gpu spec '{}' error: {}".format(gpu, e))
        return spec

    def parsed_node_selector(self) -> dict:
        sel_str = (self.task_node_selector or '').strip()
        if not sel_str:
            return dict()
        selectors = {}
        entrys = sel_str.split(',')
        for entry in entrys:
            entry = entry.strip()
            if not entry:
                continue
            kv = entry.split('=')
            if len(kv) != 2:
                continue
            k = kv[0].strip()
            v = kv[1]
            if not k:
                continue
            selectors[k] = v
        return selectors

    def get_user_path(self):
        ref_path = (self.pack_path or self.export_path or '').strip()
        if not ref_path:
            return None
        ref_path = os.path.abspath(ref_path)
        user_name = (self.creator or '').strip()
        if not user_name:
            return None

        fs = ref_path.split(os.path.sep)
        if not fs:
            return None
        pos = None
        for i, f in enumerate(fs):
            if f == user_name:
                pos = i
                break
        if pos is None:
            return None
        user_path = os.path.sep.join(fs[:pos+1])
        return os.path.normpath(user_path)

    @classmethod
    def get_context(cls):
        context = KFJobContext()
        for n in cls.__PIPELINE_ENVS:
            env_name = "PIPELINE_" + n.upper()
            env_val = os.environ.get(env_name)
            context.__setattr__(n, env_val)

        for n in cls.__KFJ_ENVS:
            env_name = "KFJ_" + n.upper()
            env_val = os.environ.get(env_name)
            context.__setattr__(n, env_val)

        return context


class JobSpec(dict):
    def is_skiped(self):
        return self.get('skip', False)

    def add_archive_file(self, src_file, archive_path, compress=False):
        new_archive = {'src': src_file, 'path_name': archive_path, 'compress': compress}
        archive_config = self.get('archive')
        if not archive_config:
            archive_config = [new_archive]
        elif isinstance(archive_config, list):
            archive_config.append(new_archive)
        else:
            archive_config = [archive_config, new_archive]
        self['archive'] = archive_config

    def add_files_to_clean(self, files):
        clean_files = self.get('clean_files')
        if not clean_files:
            clean_files = files
        else:
            if isinstance(clean_files, list):
                if isinstance(files, list):
                    clean_files.extend(files)
                else:
                    clean_files.append(files)
            else:
                if isinstance(files, list):
                    clean_files = [clean_files] + files
                else:
                    clean_files = [clean_files, files]
        self['clean_files'] = clean_files

    @property
    def name(self):
        return self.get('name')

    @property
    def namespace(self):
        return self.get('namespace')

    @property
    def output_file(self):
        return self.get('output_file')


class JobComponentEntry(object):
    def __init__(self, job_name, args=None, job_modifier=None, **job_modifier_kwargs):
        self.job_name = job_name
        self._entered = False
        self._args = args
        self._job_modifier = job_modifier
        self._job_modifier_kwargs = job_modifier_kwargs
        self._on_finish_callbacks_stack = []

    def register_finish_callback(self, func, *args, **kwargs):
        self._on_finish_callbacks_stack.append((func, args, kwargs))

    def __enter__(self):
        if self._entered:
            return self
        import argparse
        from .utils import recur_expand_param

        arg_parser = argparse.ArgumentParser(description=self.job_name)
        arg_parser.add_argument('--job', type=str, required=True, help="任务json")
        arg_parser.add_argument('--pack-path', type=str, help="用户包目录（包含所有用户文件的目录）")
        arg_parser.add_argument('--upstream-output-file', type=str, help="上游输出文件（包含路径）")
        arg_parser.add_argument('--export-path', type=str, help="数据目录")
        arg_parser.add_argument('--pipeline-id', type=str, help="pipeline id")
        arg_parser.add_argument('--run-id', type=str, help="运行id，标识每次运行")
        arg_parser.add_argument('--creator', type=str, help="pipeline的创建者")
        arg_parser.add_argument('--output-file', type=str, help="job运行完成后输出文件（不需要包含路径，会生成在数据目录下）")

        self._parsed_args = arg_parser.parse_args(self._args)
        if self._parsed_args.job[0]=='\'' or self._parsed_args.job[0]=='\"':
            self._parsed_args.job = self._parsed_args.job[1:-1]
        self._env_ctx = KFJobContext.get_context()

        job = json.loads(self._parsed_args.job)

        # update run-id when using nni search 
        if job.get('job_detail') is not None:
            nni_using = job['job_detail'].get('nni_using')
            if nni_using is not None:
                H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                nni_prefix = 'nni-{}-'.format(''.join([random.choice(H) for _ in range(6)]))
                if self._parsed_args.run_id is not None:
                    self._parsed_args.run_id = (nni_prefix + self._parsed_args.run_id)[:60]
                if self._env_ctx.run_id is not None:
                    self._env_ctx.run_id = (nni_prefix + self._env_ctx.run_id)[:60]
                print('nni search run_id update:', self._parsed_args.run_id, self._env_ctx.run_id)
        
        print("{} args: {}".format(self.job_name, self._parsed_args))
        print("ctx: {}".format(self._env_ctx))

        expand_job = recur_expand_param(job, self.data_path, self.pack_path)
        
        # update best param when setting best param path
        if expand_job.get('job_detail') is not None:
            best_param_path = expand_job['job_detail'].get('best_param_path')
            if best_param_path is not None:
                expand_job = parse_best_parameter(best_param_path, expand_job)
        
        if callable(self._job_modifier):
            expand_job = self._job_modifier(expand_job, self.pack_path, self.data_path, **self._job_modifier_kwargs)
        self._job_spec = JobSpec(expand_job)
        print("{} expanded job spec: {}".format(self.job_name, self._job_spec))
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._entered:
            return
        self._entered = False
        if exc_type is not None:
            raise
        archive_config = self.job.get('archive')
        if archive_config:
            from .utils import try_archive_by_config
            try_archive_by_config(archive_config, self.data_path, self.pack_path)

        while self._on_finish_callbacks_stack:
            on_finish, args, kwargs = self._on_finish_callbacks_stack.pop(0)
            if callable(on_finish):
                try:
                    on_finish(*args, **kwargs)
                    print("{}: called on finish callback {} with params args={}, kwargs={}"
                          .format(self.job_name, on_finish, args, kwargs))
                except Exception as e:
                    print("{}: WARNING: call on finish callback {} with params args={}, kwargs={} error: {}\n{}"
                          .format(self.job_name, on_finish, args, kwargs, e, traceback.format_exc()))

        clean_files = self.job.get('clean_files')
        if clean_files:
            from .datapath_cleaner import DataPathCleaner
            DataPathCleaner().clean(clean_files)

    @property
    def context(self) -> KFJobContext:
        return self._env_ctx

    @property
    def job(self) -> JobSpec:
        return self._job_spec

    @property
    def pack_path(self) -> str:
        path = self._parsed_args.pack_path or self._env_ctx.pack_path or './'
        path = os.path.abspath(path)
        if platform.system() == 'Windows':
            path = path.replace('\\', '/')
        return path

    @property
    def data_path(self) -> str:
        path = self._parsed_args.export_path or self._env_ctx.export_path or './'
        path = os.path.abspath(path)
        if platform.system() == 'Windows':
            path = path.replace('\\', '/')
        return path

    @property
    def export_path(self) -> str:
        return self.data_path

    @property
    def upstream_output_file(self) -> str:
        from .utils import make_abs_or_data_path
        uof = self._parsed_args.upstream_output_file or ''
        if not uof.strip():
            return ''
        return make_abs_or_data_path(uof.strip(), self.data_path, self.pack_path)

    @property
    def pipeline_id(self) -> str:
        return self._parsed_args.pipeline_id or self._env_ctx.pipeline_id

    @property
    def pipeline_name(self) -> str:
        return self.context.pipeline_name

    @property
    def task_id(self) -> str:
        return self.context.task_id

    @property
    def task_name(self) -> str:
        return self.context.task_name

    @property
    def run_id(self) -> str:
        return self._parsed_args.run_id or self._env_ctx.run_id

    @property
    def creator(self) -> str:
        return self._parsed_args.creator or self._env_ctx.creator

    @property
    def runner(self) -> str:
        return self.context.runner

    @property
    def output_file(self) -> str:
        return self._parsed_args.output_file


class JobComponentRunner(object):
    def __init__(self, job_name, output_file=None, args=None):
        self._job_name = job_name
        self.output_file = output_file
        self._args = args

    def job_func(self, jc_entry: JobComponentEntry):
        raise NotImplementedError("subclass must implement this method")

    def on_parsed_job(self, job: dict, pack_path: str, data_path: str) -> dict:
        return job

    def on_finish(self, jc_entry: JobComponentEntry, outputs):
        pass

    @property
    def job_name(self) -> str:
        return self._job_name

    def run(self):
        with JobComponentEntry(self.job_name, self._args, self.on_parsed_job) as jc_entry:
            if not jc_entry.job:
                print("{}: empty job".format(self.job_name))
                return
            if jc_entry.job.is_skiped():
                print("{}: skipped".format(self.job_name))
                return
            data_path = os.path.abspath(jc_entry.data_path)
            if not os.path.isdir(data_path):
                os.makedirs(data_path, exist_ok=True)
                print("{}: created data path '{}'".format(self.job_name, data_path))
            outputs = self.job_func(jc_entry)
            output_file = jc_entry.output_file or jc_entry.job.output_file or self.output_file
            if outputs and output_file:
                output_file = os.path.basename(os.path.abspath(output_file))
                output_file = os.path.join(jc_entry.data_path, output_file)
                output_dir = os.path.dirname(output_file)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print("{}: created output dir '{}'".format(self.job_name, output_dir))
                with open(output_file, 'w') as f:
                    if isinstance(outputs, str):
                        f.write(outputs)
                    else:
                        f.write(json.dumps(outputs))
                print("{}: wrote outputs into '{}': {}".format(self.job_name, output_file, outputs))

            jc_entry.register_finish_callback(self.on_finish, jc_entry, outputs)
