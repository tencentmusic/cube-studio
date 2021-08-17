from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import re
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob
from myapp.models.model_team import Project,Project_User
from flask_appbuilder.actions import action
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for
from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith,FilterEqual,FilterNotEqual
from wtforms.validators import EqualTo,Length
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.security.decorators import has_access
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db,event_logger
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Template
from jinja2 import contextfilter
from jinja2 import Environment, BaseLoader, DebugUndefined, StrictUndefined
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from myapp.forms import JsonValidator
from myapp.views.view_task import Task_ModelView
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList

from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget,BS3TextAreaFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
from flask_wtf.file import FileField
import shlex
import re,copy
from kubernetes.client.models import (
    V1Container, V1EnvVar, V1EnvFromSource, V1SecurityContext, V1Probe,
    V1ResourceRequirements, V1VolumeDevice, V1VolumeMount, V1ContainerPort,
    V1Lifecycle, V1Volume,V1SecurityContext
)
from .baseApi import (
    MyappModelRestApi
)
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from myapp import security_manager
import kfp    # 使用自定义的就要把pip安装的删除了
from werkzeug.datastructures import FileStorage
from kubernetes import client as k8s_client
from .base import (
    api,
    BaseMyappView,
    check_ownership,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger


class Pipeline_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        join_projects_id = security_manager.get_join_projects_id(db.session)
        # public_project_id =
        # logging.info(join_projects_id)
        return query.filter(
            or_(
                self.model.project_id.in_(join_projects_id),
                # self.model.project.name.in_(['public'])
            )
        )




from sqlalchemy.exc import InvalidRequestError,OperationalError

# 将定义pipeline的流程
# @pysnooper.snoop(watch_explode=())
def dag_to_pipeline(pipeline,dbsession):
    if not pipeline.id:
        return

    pipeline.dag_json = pipeline.fix_dag_json(dbsession)
    dbsession.commit()
    dag = json.loads(pipeline.dag_json)

    # 如果dag为空，就直接退出
    if not dag:
        return None

    all_tasks = {}
    for task_name in dag:
        # 使用临时连接，避免连接中断的问题
        # try:
            # db.session().ping()
            task = dbsession.query(Task).filter_by(name=task_name, pipeline_id=pipeline.id).first()
            if not task:
                raise MyappException('task %s not exist ' % task_name)
            all_tasks[task_name]=task

    all_ops = {}

    # 渲染字符串模板变量
    # @pysnooper.snoop()
    def template_str(src_str):
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
        des_str = rtemplate.render(creator=pipeline.created_by.username,
                                   datetime=datetime,
                                   runner=g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
                                   uuid = uuid,
                                   pipeline_id=pipeline.id,
                                   pipeline_name=pipeline.name
                                   )
        return des_str

    pipeline_global_env = template_str(pipeline.global_env.strip()) if pipeline.global_env else ''   # 优先渲染，不然里面如果有date就可能存在不一致的问题
    pipeline_global_env = [ env.strip() for env in pipeline_global_env.split('\n') if '=' in env.strip()]
    global_envs = json.loads(template_str(json.dumps(conf.get('GLOBAL_ENV', {}),indent=4,ensure_ascii=False)))
    for env in pipeline_global_env:
        key,value = env[:env.index('=')],env[env.index('=')+1:]
        global_envs[key]=value

    # @pysnooper.snoop()
    def get_ops(task_name):
        task = all_tasks[task_name]
        ops_args = []
        task_args = json.loads(task.args)
        for task_attr_name in task_args:
            # 添加参数名
            if type(task_args[task_attr_name])==bool:
                if task_args[task_attr_name]:
                    ops_args.append('%s' % str(task_attr_name))
            # 添加参数值
            elif type(task_args[task_attr_name])==dict or type(task_args[task_attr_name])==list:
                ops_args.append('%s' % str(task_attr_name))
                ops_args.append('%s' % json.dumps(task_args[task_attr_name],ensure_ascii=False))
            elif not task_args[task_attr_name]:     # 如果参数值为空，则都不添加
                pass
            else:
                ops_args.append('%s' % str(task_attr_name))
                ops_args.append('%s'%str(task_args[task_attr_name]))   # 这里应该对不同类型的参数名称做不同的参数处理，比如bool型，只有参数，没有值

        # pipeline_global_args = global_env.strip().split(' ') if global_env else []
        # pipeline_global_args = [arg.strip() for arg in pipeline_global_args if arg.strip()]
        # for global_arg in pipeline_global_args:
        #     ops_args.append(global_arg)

        # 创建ops的pod的创建参数
        container_kwargs={}

        # 设置privileged
        if task.job_template.privileged:
            container_kwargs['security_context'] = V1SecurityContext(privileged=task.job_template.privileged)

        # 设置环境变量
        container_envs = []
        if task.job_template.env:
            envs = re.split('\r|\n',task.job_template.env)
            for env in envs:
                env_key,env_value = env.split('=')[0],env.split('=')[1]
                container_envs.append(V1EnvVar(env_key,env_value))

        # 设置全局环境变量
        for global_env_key in global_envs:
            container_envs.append(V1EnvVar(global_env_key,global_envs[global_env_key]))

        # 设置task的默认环境变量
        container_envs.append(V1EnvVar("KFJ_TASK_ID", str(task.id)))
        container_envs.append(V1EnvVar("KFJ_TASK_NAME", str(task.name)))
        container_envs.append(V1EnvVar("KFJ_TASK_NODE_SELECTOR", str(task.node_selector)))
        container_envs.append(V1EnvVar("KFJ_TASK_VOLUME_MOUNT", str(task.volume_mount)))
        container_envs.append(V1EnvVar("KFJ_TASK_IMAGES", str(task.job_template.images)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_CPU", str(task.resource_cpu)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_MEMORY", str(task.resource_memory)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_GPU", str(task.resource_gpu.replace("+",''))))
        container_envs.append(V1EnvVar("GPU_TYPE", os.environ.get("GPU_TYPE", "NVIDIA")))

        container_kwargs['env']=container_envs


        # 创建工作目录
        if task.job_template.workdir and task.job_template.workdir.strip():
            container_kwargs['working_dir'] = task.job_template.workdir.strip()
        if task.working_dir and task.working_dir.strip():
            container_kwargs['working_dir'] = task.working_dir.strip()


        # # 创建label，这样能让每个pod都找到运行人。
        # container_lables={
        #     'upload-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
        #     'run-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username
        # }
        # container_kwargs['labels']=container_lables


        task_command = ''
        # 不使用command形式携带host
        # if 0 and task.job_template.hostAliases:
        #     hostAliases = re.split('\r|\n',task.job_template.hostAliases)
        #     hostAliases=["echo %s >> /etc/hosts"%hostAliase for hostAliase in hostAliases if hostAliase]
        #     task_command+=' && '.join(hostAliases)

        if task.command:
            commands = re.split('\r|\n',task.command)
            commands = [command.strip() for command in commands if command.strip()]
            if task_command:
                task_command += " && " + " && ".join(commands)
            else:
                task_command += " && ".join(commands)

        job_template_entrypoint = task.job_template.entrypoint.strip().replace("  ",'') if task.job_template.entrypoint else ''


        command=None
        if job_template_entrypoint:
            command = job_template_entrypoint

        if task_command:
            command = task_command.replace("  ",'')


        # entrypoint = task.job_template.images.entrypoint
        # overwrite_entrypoint = task.overwrite_entrypoint
        # if entrypoint and entrypoint.strip() and not overwrite_entrypoint:
        #     if task_command:
        #         task_command+=" && "+entrypoint.strip()
        #     else:
        #         task_command += entrypoint.strip()

        # if not overwrite_entrypoint and (not entrypoint or not entrypoint.strip()):
        #     raise MyappException('job template %s 的镜像的入口命令未填写，联系%s添加镜像入口命令，或选择覆盖入口命令'%(task.job_template.name,task.job_template.created_by.username))

        # task_commands = re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', task_command)   # task_command.split(' ')
        # task_commands = re.split(' ',task_command)
        # task_commands = [task_command for task_command in task_commands if task_command ]
        # ops = kfp.dsl.ContainerOp(
        #     name=task.name,
        #     image=task.job_template.images.name,
        #     arguments=ops_args,
        #     command=task_commands if task_commands else None,
        #     container_kwargs=container_kwargs
        # )

        if task.job_template.name==conf.get('CUSTOMIZE_JOB'):
            ops = kfp.dsl.ContainerOp(
                name=task.name,
                image=json.loads(task.args).get('images'),
                command=['sh','-c',task.command] if task.command else None,
                container_kwargs=container_kwargs,
                file_outputs = json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None
            )
        else:

            # 数组方式
            # if task_command:
            #     task_command = task_command.split(' ')
            #     task_command = [command for command in task_command if command]
            ops = kfp.dsl.ContainerOp(
                name=task.name,
                image=task.job_template.images.name,
                arguments=ops_args,
                command=command.split(' ') if command else None,
                container_kwargs=container_kwargs,
                file_outputs=json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None
            )

            # 合并方式
            # ops = kfp.dsl.ContainerOp(
            #     name=task.name,
            #     image=task.job_template.images.name,
            #     arguments=ops_args,
            #     command=['sh', '-c', task_command] if task_command else None,
            #     container_kwargs=container_kwargs,
            #     file_outputs=json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None
            # )

        # 添加用户自定义挂载
        task.volume_mount=task.volume_mount.strip() if task.volume_mount else ''
        if task.volume_mount:
            volume_mounts = re.split(',|;',task.volume_mount)
            for volume_mount in volume_mounts:
                volume,mount = volume_mount.split(":")[0].strip(),volume_mount.split(":")[1].strip()
                if "(pvc)" in volume:
                    pvc_name = volume.replace('(pvc)','').replace(' ','')
                    temps = re.split('_|\.|/', pvc_name)
                    temps = [temp for temp in temps if temp]
                    volumn_name = '-'.join(temps)
                    ops=ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                           persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)))\
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=os.path.join(mount,task.pipeline.created_by.username), name=volumn_name,sub_path=task.pipeline.created_by.username))
                if "(hostpath)" in volume:
                    hostpath_name = volume.replace('(hostpath)', '').replace(' ', '')
                    temps = re.split('_|\.|/', hostpath_name)
                    temps = [temp for temp in temps if temp]
                    volumn_name = '-'.join(temps)

                    ops = ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                             host_path=k8s_client.V1HostPathVolumeSource(path=hostpath_name))) \
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount, name=volumn_name))
                if "(configmap)" in volume:
                    configmap_name = volume.replace('(configmap)', '').replace(' ', '')
                    temps = re.split('_|\.|/', configmap_name)
                    temps = [temp for temp in temps if temp]
                    volumn_name = '-'.join(temps)

                    ops = ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                             config_map=k8s_client.V1ConfigMapVolumeSource(name=configmap_name))) \
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount, name=volumn_name))

                if "(memory)" in volume:
                    memory_size = volume.replace('(memory)', '').replace(' ', '').lower().replace('g','')
                    volumn_name = 'memory-%s'%memory_size
                    ops = ops.add_volume(k8s_client.V1Volume(name=volumn_name,empty_dir=k8s_client.V1EmptyDirVolumeSource(medium='Memory',size_limit='%sGi'%memory_size)))\
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount, name=volumn_name))



        # 添加上游依赖
        if "upstream" in dag[task_name] and dag[task_name]['upstream']:
            upstream_tasks = dag[task_name]['upstream']
            if type(upstream_tasks)==dict:
                upstream_tasks=list(upstream_tasks.keys())
            if type(upstream_tasks)==str:
                upstream_tasks = upstream_tasks.split(',;')
            if type(upstream_tasks)!=list:
                raise MyappException('%s upstream is not valid'%task_name)
            for upstream_task in upstream_tasks:   # 可能存在已删除的upstream_task，这个时候是不添加的
                add_upstream=False
                for task1_name in all_ops:
                    if task1_name==upstream_task:
                        ops.after(all_ops[task1_name])  # 配置任务顺序
                        add_upstream=True
                        break
                if not add_upstream:
                    raise MyappException('%s upstream %s is not exist' % (task_name,upstream_task))

        # 添加node selector
        if task.node_selector:
            for selector in re.split(',|;|\n|\t', str(task.node_selector)):
                ops.add_node_selector_constraint(selector.split('=')[0].strip(),selector.split('=')[1].strip())


        # 根据用户身份设置机器选择器
        if g and g.user and g.user.org:
            ops.add_node_selector_constraint('org-'+g.user.org,'true')
        elif pipeline.created_by.org:  # 对定时任务，没有g存储
            ops.add_node_selector_constraint('org-'+pipeline.created_by.org,'true')

        # 设置host
        # if task.job_template.hostAliases:
        #     hostAliases = re.split('\r|\n',task.job_template.hostAliases)
        #     hostAliases=["echo %s >> /etc/hosts"%hostAliase for hostAliase in hostAliases if hostAliase]
        #     task_command+=' && '.join(hostAliases)

        # 添加pod label
        ops.add_pod_label("pipeline-id",str(pipeline.id))
        ops.add_pod_label("pipeline-name", str(pipeline.name))
        ops.add_pod_label("task-name", str(task.name))
        ops.add_pod_label("run-id", global_envs.get('KFJ_RUN_ID',''))
        ops.add_pod_label("task-id", str(task.id))
        ops.add_pod_label("task-id", str(task.id))
        ops.add_pod_label('upload-rtx', g.user.username if g and g.user and g.user.username else pipeline.created_by.username)
        ops.add_pod_label('run-rtx', g.user.username if g and g.user and g.user.username else pipeline.created_by.username)
        ops.add_pod_label('pipeline-rtx', pipeline.created_by.username)

        # 设置重试次数
        if task.retry:
            ops.set_retry(int(task.retry))

        # 设置超时
        if task.timeout:
            ops.set_timeout(int(task.timeout))

        resource_cpu = task.job_template.get_env('TASK_RESOURCE_CPU') if task.job_template.get_env('TASK_RESOURCE_CPU') else task.resource_cpu
        resource_gpu = task.job_template.get_env('TASK_RESOURCE_GPU') if task.job_template.get_env('TASK_RESOURCE_GPU') else task.resource_gpu
        resource_memory = task.job_template.get_env('TASK_RESOURCE_MEMORY') if task.job_template.get_env('TASK_RESOURCE_MEMORY') else task.resource_memory

        # 设置资源限制
        if resource_memory:
            if not '~' in resource_memory:
                ops.set_memory_request(resource_memory)
                ops.set_memory_limit(resource_memory)
            else:
                # logging.info(task.resource_memory)
                ops.set_memory_request(resource_memory.split("~")[0])
                ops.set_memory_limit(resource_memory.split("~")[1])

        if resource_cpu:
            if not '~' in resource_cpu:
                ops.set_cpu_request(resource_cpu)
                ops.set_cpu_limit(resource_cpu)
            else:
                # logging.info(task.resource_cpu)
                ops.set_cpu_request(resource_cpu.split("~")[0])
                ops.set_cpu_limit(resource_cpu.split("~")[1])

        if resource_gpu:
            gpu_type = conf.get('GPU_TYPE','NVIDIA')
            if gpu_type=='NVIDIA':
                if resource_gpu and core.get_gpu(resource_gpu)[0]>0:
                    ops.set_gpu_limit(core.get_gpu(resource_gpu)[0])
            if gpu_type=='TENCENT':
                gpu_core,gpu_mem = core.get_gpu(resource_gpu)
                if gpu_core and gpu_mem:
                    ops.add_resource_request('tencent.com/vcuda-core',str(gpu_core))
                    ops.add_resource_request('tencent.com/vcuda-memory', str(4*gpu_mem))
                    ops.add_resource_limit('tencent.com/vcuda-core',str(gpu_core))
                    ops.add_resource_limit("tencent.com/vcuda-memory", str(4*gpu_mem))


        all_ops[task_name]=ops

    # 这里面是真正的pipeline name  上传时指定的是version的name
    @kfp.dsl.pipeline(name=pipeline.name,description=pipeline.describe)
    def my_pipeline():
        for task_name in dag:
            get_ops(task_name)



    # pipeline运行的相关配置
    pipeline_conf = kfp.dsl.PipelineConf()
    hubsecret_list = []
    for task_name in all_tasks:
        # 配置拉取秘钥。本来在contain里面，workflow在外面
        task_temp = all_tasks[task_name]
        if task_temp.job_template.images.repository.hubsecret:
            hubsecret =  task_temp.job_template.images.repository.hubsecret
            if hubsecret not in hubsecret_list:
                hubsecret_list.append(hubsecret)
                pipeline_conf.image_pull_secrets.append(k8s_client.V1LocalObjectReference(name=hubsecret))

        # 配置host
        if task_temp.job_template.hostAliases:
            hostAliases_list = re.split('\r|\n', task_temp.job_template.hostAliases)
            for row in hostAliases_list:
                hosts = row.split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    pipeline_conf.set_host_aliases(ip=hosts[0],hostnames=hosts[1:])


    # 配置默认拉取策略
    if pipeline.image_pull_policy:
        pipeline_conf.image_pull_policy = pipeline.image_pull_policy


    # 设置默认机器选择器
    if pipeline.node_selector:
        for selector in re.split(',|;|\n|\t', str(pipeline.node_selector)):
            pipeline_conf.set_default_pod_node_selector(selector.split('=')[0].strip(),selector.split('=')[1].strip())


    # 根据用户身份设置机器选择器
    if g and g.user and g.user.org:
        pipeline_conf.set_default_pod_node_selector('org-'+g.user.org,'true')
    elif pipeline.created_by.org:  # 对定时任务，没有g存在
        pipeline_conf.set_default_pod_node_selector('org-'+pipeline.created_by.org,'true' )

    # 设置并发
    if pipeline.parallelism:
        pipeline_conf.parallelism = int(pipeline.parallelism)


    # 设置workflow标签
    # if pipeline._extra_data['upload_pipeline']:
    pipeline_conf.labels['upload-rtx']=g.user.username if g and g.user and g.user.username else pipeline.created_by.username
    pipeline_conf.labels['run-rtx'] = g.user.username if g and g.user and g.user.username else pipeline.created_by.username
    pipeline_conf.labels['pipeline-rtx'] = pipeline.created_by.username
    pipeline_conf.labels['save-time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    pipeline_conf.labels['pipeline-id'] = str(pipeline.id)
    pipeline_conf.labels['run-id'] = global_envs.get('KFJ_RUN_ID','')   # 以此来绑定运行时id，不能用kfp的run—id。那个是传到kfp以后才产生的。


    kfp.compiler.Compiler().compile(my_pipeline, pipeline.name+'.yaml',pipeline_conf=pipeline_conf)
    file = open(pipeline.name+'.yaml',mode='rb')
    pipeline_file = template_str(str(file.read(),encoding='utf-8'))
    file.close()
    return pipeline_file
    # logging.info(pipeline.pipeline_file)


# @pysnooper.snoop(watch_explode=())
def upload_pipeline(pipeline):

    if not pipeline.pipeline_file:
        return None,None

    file = open(pipeline.name + '.yaml', mode='wb')
    file.write(bytes(pipeline.pipeline_file,encoding='utf-8'))
    file.close()
    client = kfp.Client(pipeline.project.cluster.get('KFP_HOST'))
    pipeline_argo = None
    if pipeline.pipeline_argo_id:
        try:
            pipeline_argo = client.get_pipeline(pipeline.pipeline_argo_id)
        except Exception as e:
            logging.error(e)

    if pipeline_argo:
        pipeline_argo_version = client.upload_pipeline_version(pipeline_package_path=pipeline.name + '.yaml', pipeline_version_name=pipeline.name+"_version_at_"+datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),pipeline_id=pipeline_argo.id)
        time.sleep(1)   # 因为创建是异步的，要等k8s反应，所以有时延
        return pipeline_argo.id,pipeline_argo_version.id
    else:
        exist_pipeline_argo_id = None
        try:
            exist_pipeline_argo_id = client.get_pipeline_id(pipeline.name)
        except Exception as e:
            logging.error(e)

        if exist_pipeline_argo_id:
            pipeline_argo_version = client.upload_pipeline_version(pipeline_package_path=pipeline.name + '.yaml',pipeline_version_name=pipeline.name + "_version_at_" + datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),pipeline_id=exist_pipeline_argo_id)
            time.sleep(1)
            return exist_pipeline_argo_id,pipeline_argo_version.id
        else:
            pipeline_argo = client.upload_pipeline(pipeline.name + '.yaml', pipeline_name=pipeline.name)
            time.sleep(1)
            return pipeline_argo.id,pipeline_argo.default_version.id



# @pysnooper.snoop(watch_explode=())
def run_pipeline(pipeline):
    # logging.info(pipeline)
    # return
    # 如果没值就先upload
    if not pipeline.pipeline_argo_id or not pipeline.version_id:
        pipeline_argo_id,pipeline_argo_version_id = upload_pipeline(pipeline)   # 必须上传新版本
    else:
        pipeline_argo_id = pipeline.pipeline_argo_id
        pipeline_argo_version_id = pipeline.version_id

    client = kfp.Client(pipeline.project.cluster.get('KFP_HOST'))
    # 先创建一个实验，在在这个实验中运行指定pipeline
    experiment=None
    try:
        experiment = client.get_experiment(experiment_name=pipeline.name)
    except Exception as e:
        logging.error(e)
    if not experiment:
        try:
            experiment = client.create_experiment(name=pipeline.name,description=pipeline.name)  # 现在要求describe不能是中文了
        except Exception as e:
            print(e)
            return None,None,None
    # 直接使用pipeline最新的版本运行
    try:
        run = client.run_pipeline(experiment_id = experiment.id,pipeline_id=pipeline_argo_id,version_id=pipeline_argo_version_id,job_name=pipeline.name+"_version_at_"+datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
        return pipeline_argo_id, pipeline_argo_version_id, run.id
    except Exception as e:
        print(e)
        raise e





class Pipeline_ModelView_Base():
    label_title='任务流'
    datamodel = SQLAInterface(Pipeline)
    check_redirect_list_url = '/pipeline_modelview/list/?_flt_2_name='
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    base_permissions = ['can_show','can_list','can_delete','can_add']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']

    list_columns = ['id','project','pipeline_url','creator','modified']
    add_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','global_env','alert_status','alert_user']
    show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','global_env','dag_json_html','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand_html']
    edit_columns = add_columns
    # add_form_query_rel_fields = {
    #     "project": [["name", Project_Filter, 'org']]
    # }
    # edit_form_query_rel_fields = add_form_query_rel_fields

    base_filters = [["id", Pipeline_Filter, lambda: []]]  # 设置权限过滤器
    conv = GeneralModelConverter(datamodel)


    # 获取查询自己所在的项目组的project
    def filter_project():
        query = db.session.query(Project)
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return query.filter(Project.type=='org').order_by(Project.id.desc())

        # 查询自己拥有的项目
        my_user_id = g.user.get_id() if g.user else 0
        owner_ids_query = db.session.query(Project_User.project_id).filter(Project_User.user_id == my_user_id)

        return query.filter(Project.id.in_(owner_ids_query)).filter(Project.type=='org').order_by(Project.id.desc())


    add_form_extra_fields = {

        "name": StringField(
            _(datamodel.obj.lab('name')),
            description="英文名(字母、数字、- 组成)，最长50个字符",
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54),DataRequired()]
        ),
        "project":QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_project,
            allow_blank=True,
            widget=Select2Widget()
        ),
        "dag_json": StringField(
            _(datamodel.obj.lab('dag_json')),
            widget=MyBS3TextAreaFieldWidget(rows=10),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "namespace": StringField(
            _(datamodel.obj.lab('namespace')),
            description="部署task所在的命名空间(目前无需填写)",
            default='pipeline',
            widget=BS3TextFieldWidget()
        ),
        "node_selector": StringField(
            _(datamodel.obj.lab('node_selector')),
            description="部署task所在的机器(目前无需填写)",
            widget=BS3TextFieldWidget(),
            default=datamodel.obj.node_selector.default.arg
        ),
        "image_pull_policy": SelectField(
            _(datamodel.obj.lab('image_pull_policy')),
            description="镜像拉取策略(always为总是拉取远程镜像，IfNotPresent为若本地存在则使用本地镜像)",
            widget=Select2Widget(),
            choices=[['Always','Always'],['IfNotPresent','IfNotPresent']]
        ),
        "parallelism": IntegerField(
            _(datamodel.obj.lab('parallelism')),
            description="pipeline中可同时运行的task数目",
            widget=BS3TextFieldWidget(),
            default=3,
            validators=[DataRequired()]
        ),
        "global_env": StringField(
            _(datamodel.obj.lab('global_env')),
            description="公共环境变量会以环境变量的形式传递给每个task，可以配置多个公共环境变量，每行一个，支持datetime/creator/runner/uuid/pipeline_id等变量 例如：USERNAME={{creator}}",
            widget=BS3TextAreaFieldWidget()
        ),
        "schedule_type":SelectField(
            _(datamodel.obj.lab('schedule_type')),
            description="调度类型，once仅运行一次，crontab周期运行，crontab配置保存一个小时候后才生效",
            widget=Select2Widget(),
            choices=[['once','once'],['crontab','crontab']]
        ),
        "cron_time": StringField(
            _(datamodel.obj.lab('cron_time')),
            description="周期任务的时间设定 * * * * * 表示为 minute hour day month week",
            widget=BS3TextFieldWidget()
        ),
        "alert_status":MySelectMultipleField(
            label=_(datamodel.obj.lab('alert_status')),
            widget=Select2ManyWidget(),
            choices=[[x, x] for x in
                     ['Created', 'Pending', 'Running', 'Succeeded', 'Failed', 'Unknown', 'Waiting', 'Terminated']],
            description="选择通知状态"
        ),
        "alert_user": StringField(
            label=_(datamodel.obj.lab('alert_user')),
            widget=BS3TextFieldWidget(),
            description="选择通知用户，每个用户使用逗号分隔"
        )

    }


    edit_form_extra_fields = add_form_extra_fields


    related_views = [Task_ModelView, ]

    # 验证args参数
    # @pysnooper.snoop(watch_explode=('item'))
    def pipeline_args_check(self, item):
        core.validate_str(item.name,'name')
        if not item.dag_json:
            item.dag_json='{}'
        core.validate_json(item.dag_json)
        # 校验task的关系，没有闭环，并且顺序要对。没有配置的，自动没有上游，独立
        # @pysnooper.snoop()
        def order_by_upstream(dag_json):
            order_dag={}
            tasks_name = list(dag_json.keys())  # 如果没有配全的话，可能只有局部的task
            i=0
            while tasks_name:
                i+=1
                if i>100:  # 不会有100个依赖关系
                    break
                for task_name in tasks_name:
                    # 没有上游的情况
                    if not dag_json[task_name]:
                        order_dag[task_name]={}
                        tasks_name.remove(task_name)
                        continue
                    # 没有上游的情况
                    elif 'upstream' not in dag_json[task_name] or not dag_json[task_name]['upstream']:
                        order_dag[task_name] = {}
                        tasks_name.remove(task_name)
                        continue
                    # 如果有上游依赖的话，先看上游任务是否已经加到里面了。
                    upstream_all_ready=True
                    for upstream_task_name in dag_json[task_name]['upstream']:
                        if upstream_task_name not in order_dag:
                            upstream_all_ready=False
                    if upstream_all_ready:
                        order_dag[task_name]=dag_json[task_name]
                        tasks_name.remove(task_name)
            if list(dag_json.keys()).sort()!=list(order_dag.keys()).sort():
                flash('dag pipeline 存在循环或未知上游',category='warning')
                raise MyappException('dag pipeline 存在循环或未知上游')
            return order_dag

        # 配置上缺少的默认上游
        dag_json = json.loads(item.dag_json)
        tasks = item.get_tasks(db.session)
        if tasks and dag_json:
            for task in tasks:
                if task.name not in dag_json:
                    dag_json[task.name]={
                        "upstream": []
                    }
        item.dag_json = json.dumps(order_by_upstream(copy.deepcopy(dag_json)),ensure_ascii=False,indent=4)
        # 生成workflow，如果有id，
        if item.id and item.get_tasks():
            item.pipeline_file = dag_to_pipeline(item,db.session)
        else:
            item.pipeline_file = None


        # raise Exception('args is not valid')

    # 合并上下游关系
    # @pysnooper.snoop(watch_explode=('pipeline'))
    def merge_upstream(self,pipeline):
        logging.info(pipeline)

        dag_json={}
        # 根据参数生成args字典。一层嵌套的形式
        for arg in pipeline.__dict__:
            if len(arg)>5 and arg[:5] == 'task.':
                task_upstream = getattr(pipeline,arg)
                dag_json[arg[5:]]={
                    "upstream":task_upstream if task_upstream else []
                }
        if dag_json:
            pipeline.dag_json = json.dumps(dag_json)

    # @pysnooper.snoop()
    def pre_add(self, item):
        item.name = item.name.replace('_', '-')[0:54].lower()
        # item.alert_status = ','.join(item.alert_status)
        self.pipeline_args_check(item)
        item.create_datetime=datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()

    # @pysnooper.snoop()
    def pre_update(self, item):

        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        item.name = item.name.replace('_', '-')[0:54].lower()
        # item.alert_status = ','.join(item.alert_status)
        self.merge_upstream(item)
        self.pipeline_args_check(item)
        item.change_datetime = datetime.datetime.now()

    def pre_update_get(self,item):
        item.dag_json = item.fix_dag_json()
        item.expand = json.dumps(item.fix_expand(),indent=4,ensure_ascii=False)
        db.session.commit()

    # 删除前先把下面的task删除了
    # @pysnooper.snoop()
    def pre_delete(self, pipeline):
        tasks = pipeline.get_tasks()
        for task in tasks:
            db.session.delete(task)
        db.session.commit()


    @event_logger.log_this
    @expose("/list/")
    @has_access
    def list(self):
        args = request.args.to_dict()
        if '_flt_0_created_by' in args and args['_flt_0_created_by']=='':
            print(request.url)
            print(request.path)
            flash('去除过滤条件->查看所有pipeline','success')
            return redirect(request.url.replace('_flt_0_created_by=','_flt_0_created_by=%s'%g.user.id))

        widgets = self._list()
        res = self.render_template(
            self.list_template, title=self.list_title, widgets=widgets
        )
        return res


    # @event_logger.log_this
    @action(
        "download", __("Download"), __("Download Yaml"), "fa-download", multiple=False, single=True
    )
    def download(self, item):
        file_name = item.name+'-download.yaml'
        file_dir = os.path.join(conf.get('DOWNLOAD_FOLDER'),'pipeline')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file = open(os.path.join(file_dir,file_name), mode='wb')
        pipeline_file = item.pipeline_file
        try:
            import yaml
            pipeline_yaml = yaml.safe_load(pipeline_file)
            pipeline_yaml['metadata']['name']=item.name+'-'+uuid.uuid4().hex[:4]
            pipeline_yaml['metadata']['namespace'] = 'pipeline'
            pipeline_file = yaml.safe_dump(pipeline_yaml)
        except Exception as e:
            print(e)

        file.write(bytes(pipeline_file, encoding='utf-8'))
        file.close()
        response = make_response(send_from_directory(file_dir, file_name, as_attachment=True, conditional=True))

        response.headers[
            "Content-Disposition"
        ] = f"attachment; filename={file_name}"
        logging.info("Ready to return response")
        return response


    # 获取当期运行时workfflow的数量
    def get_running_num(self, pipeline):
        back_crds = []
        try:
            k8s_client = py_k8s.K8s(pipeline.project.cluster['KUBECONFIG'])
            crd_info = conf.get("CRD_INFO", {}).get('workflow', {})
            if crd_info:
                crds = k8s_client.get_crd(group=crd_info['group'], version=crd_info['version'],plural=crd_info['plural'], namespace=pipeline.namespace)
                for crd in crds:
                    if crd.get('labels','{}'):
                        labels = json.loads(crd['labels'])
                        if labels.get('pipeline-id','')==str(pipeline.id):
                            back_crds.append(crd)
            return back_crds
        except Exception as e:
            print(e)
        return back_crds

    # @pysnooper.snoop()
    def delete_not_running_crd(self,crds):

        for crd in crds:
            try:
                db_crd = db.session.query(Workflow).filter_by(name=crd['name']).first()
                pipeline = db_crd.pipeline
                if pipeline:
                    k8s_client = py_k8s.K8s(pipeline.project.cluster['KUBECONFIG'])
                else:
                    k8s_client = py_k8s.K8s()

                k8s_client.delete_workflow(
                    all_crd_info = conf.get("CRD_INFO", {}),
                    namespace=crd['namespace'],
                    run_id = json.loads(crd['labels']).get("run-id",'')
                )
                if db_crd:
                    db_crd.status='Deleted'
                    db.session.commit()
            except Exception as e:
                print(e)


    def check_pipeline_perms(user_fun):
        # @pysnooper.snoop()
        def wraps(*args, **kwargs):
            pipeline_id = int(kwargs.get('pipeline_id','0'))
            if not pipeline_id:
                response = make_response("pipeline_id not exist")
                response.status_code = 404
                return response

            user_roles = [role.name.lower() for role in g.user.roles]
            if "admin" in user_roles:
                return user_fun(*args, **kwargs)

            join_projects_id = security_manager.get_join_projects_id(db.session)
            pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
            if pipeline.project.id in join_projects_id:
                return user_fun(*args, **kwargs)

            response = make_response("no perms to run pipeline %s"%pipeline_id)
            response.status_code = 403
            return response

        return wraps


    # # @event_logger.log_this
    @expose("/run_pipeline/<pipeline_id>", methods=["GET", "POST"])
    @check_pipeline_perms
    def run_pipeline(self,pipeline_id):
        print(pipeline_id)
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        time.sleep(1)
        # if pipeline.changed_on+datetime.timedelta(seconds=5)>datetime.datetime.now():
        #     flash("发起运行实例，太过频繁，5s后重试",category='warning')
        #     return redirect('/pipeline_modelview/list/?_flt_2_name=')
        back_crds = self.get_running_num(pipeline)

        # 把消息加入到源数据库
        for crd in back_crds:
            try:
                workflow = db.session.query(Workflow).filter_by(name=crd['name']).first()
                if not workflow:
                    username = ''
                    labels = json.loads(workflow['labels'])
                    if 'run-rtx' in labels:
                        username = labels['run-rtx']
                    elif 'upload-rtx' in labels:
                        username = labels['upload-rtx']

                    workflow = Workflow(name=crd['name'], namespace=crd['namespace'], create_time=crd['create_time'],
                                        status=crd['status'],
                                        annotations=crd['annotations'],
                                        labels=crd['labels'],
                                        spec=crd['spec'],
                                        status_more=crd['status_more'],
                                        username=username
                                        )
                    db.session.add(workflow)
                    db.session.commit()
            except Exception as e:
                print(e)

        # 这里直接删除所有的历史任务流，正在运行的也删除掉
        not_running_crds = back_crds  # [crd for crd in back_crds if 'running' not in crd['status'].lower()]
        self.delete_not_running_crd(not_running_crds)

        # running_crds = [1 for crd in back_crds if 'running' in crd['status'].lower()]
        # if len(running_crds)>0:
        #     flash("发现当前运行实例 %s 个，目前集群仅支持每个任务流1个运行实例，若要重新发起实例，请先stop旧实例"%len(running_crds),category='warning')
        #     # run_instance = '/workflow_modelview/list/?_flt_2_name=%s'%pipeline.name.replace("_","-")[:54]
        #     run_instance = r'/workflow_modelview/list/?_flt_2_labels="pipeline-id"%3A+"'+'%s"' % pipeline_id
        #     return redirect(run_instance)


        # self.delete_workflow(pipeline)
        pipeline.pipeline_file = dag_to_pipeline(pipeline, db.session)  # 合成workflow
        # print('make pipeline file %s' % pipeline.pipeline_file)
        # return
        print('begin upload and run pipeline %s' % pipeline.name)
        pipeline.version_id = ''
        pipeline.run_id = ''
        pipeline_argo_id, version_id, run_id = run_pipeline(pipeline)   # 会根据版本号是否为空决定是否上传
        print('success upload and run pipeline %s,pipeline_argo_id %s, version_id %s,run_id %s ' % (pipeline.name, pipeline_argo_id, version_id, run_id))
        pipeline.pipeline_argo_id = pipeline_argo_id
        pipeline.version_id = version_id
        pipeline.run_id = run_id
        db.session.commit()  # 更新

        run_url = conf.get('PIPELINE_URL') + "runs/details/" + run_id
        logging.info(run_url)
        # run_url='http://www.baidu.com/http://www.baidu.com/'
        return redirect("/pipeline_modelview/web/log/%s"%pipeline_id)
        # return redirect(run_url)


    # # @event_logger.log_this
    @expose("/web/<pipeline_id>", methods=["GET"])
    def web(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        db_tasks = pipeline.get_tasks(db.session)
        if db_tasks:
            try:
                tasks={}
                for task in db_tasks:
                    tasks[task.name]=task.to_json()
                expand = core.fix_task_position(pipeline.to_json(),tasks)
                pipeline.expand=json.dumps(expand,indent=4,ensure_ascii=False)
                db.session.commit()
            except Exception as e:
                print(e)

        print(pipeline_id)
        data = {
            "url": '/static/appbuilder/vison/index.html?pipeline_id=%s'%pipeline_id  # 前后端集成完毕，这里需要修改掉
        }
        # 返回模板
        return self.render_template('link.html', data=data)


    # # @event_logger.log_this
    @expose("/web/log/<pipeline_id>", methods=["GET"])
    def web_log(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        if pipeline.run_id:
            data = {
                "url": pipeline.project.cluster.get('PIPELINE_URL') + "runs/details/" + pipeline.run_id,
                "target":"div.page_f1flacxk:nth-of-type(2)",
                "delay":1000,
                "loading": True
            }
            # 返回模板
            if pipeline.project.cluster['NAME']=='local':
                return self.render_template('link.html', data=data)
            else:
                return self.render_template('external_link.html', data=data)
        else:
            flash('no running instance','warning')
            return redirect('/pipeline_modelview/web/%s'%pipeline.id)


    # # @event_logger.log_this
    @expose("/web/pod/<pipeline_id>", methods=["GET"])
    def web_pod(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        data = {
            "url": pipeline.project.cluster.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (conf.get('PIPELINE_NAMESPACE'), pipeline.name.replace('_', '-')),
            "target":"div.kd-chrome-container.kd-bg-background",
            "delay":1000,
            "loading": True
        }
        # 返回模板
        if pipeline.project.cluster['NAME']=='local':
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)



    # # @event_logger.log_this
    @expose("/copy_pipeline/<pipeline_id>", methods=["GET", "POST"])
    # @check_pipeline_perms
    def copy_pipeline(self,pipeline_id):
        print(pipeline_id)
        message=''
        try:
            pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
            new_pipeline = pipeline.clone()
            new_pipeline.name = new_pipeline.name.replace('_', '-') + "-copy-" + uuid.uuid4().hex[:4]
            new_pipeline.created_on = datetime.datetime.now()
            new_pipeline.changed_on = datetime.datetime.now()
            db.session.add(new_pipeline)
            db.session.commit()
            # 复制绑定的task，并绑定新的pipeline
            for task in pipeline.get_tasks():
                new_task = task.clone()
                new_task.pipeline_id = new_pipeline.id
                new_task.create_datetime = datetime.datetime.now()
                new_task.change_datetime = datetime.datetime.now()
                db.session.add(new_task)
                db.session.commit()
            return jsonify(new_pipeline.to_json())
            # return redirect('/pipeline_modelview/web/%s'%new_pipeline.id)
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            message=str(e)
        response = make_response("copy pipeline %s error: %s" % (pipeline_id,message))
        response.status_code = 500
        return response

    @action(
        "copy", __("Copy Pipeline"), confirmation=__('Copy Pipeline'), icon="fa-copy",multiple=True, single=False
    )
    def copy(self, pipelines):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        try:
            for pipeline in pipelines:
                new_pipeline = pipeline.clone()
                new_pipeline.name = new_pipeline.name.replace('_','-')+"-copy-"+uuid.uuid4().hex[:4]
                new_pipeline.created_on = datetime.datetime.now()
                new_pipeline.changed_on = datetime.datetime.now()
                db.session.add(new_pipeline)
                db.session.commit()
                # 复制绑定的task，并绑定新的pipeline
                for task in pipeline.get_tasks():
                    new_task = task.clone()
                    new_task.pipeline_id = new_pipeline.id
                    new_task.create_datetime = datetime.datetime.now()
                    new_task.change_datetime = datetime.datetime.now()
                    db.session.add(new_task)
                    db.session.commit()
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            raise e

        return redirect(request.referrer)


class Pipeline_ModelView(Pipeline_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Pipeline)
    # base_order = ("changed_on", "desc")
    # order_columns = ['changed_on']


appbuilder.add_view(Pipeline_ModelView,"任务流",href="/pipeline_modelview/list/?_flt_0_created_by=",icon = 'fa-sitemap',category = '训练')

# 添加api
class Pipeline_ModelView_Api(Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Pipeline)
    route_base = '/pipeline_modelview/api'
    show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','global_env','dag_json','pipeline_file','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    # show_columns = ['name','describe','project','dag_json','namespace','global_env','schedule_type','cron_time','pipeline_file','pipeline_argo_id','version_id','run_id','node_selector','image_pull_policy','parallelism','alert_status']
    list_columns = show_columns
    add_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','image_pull_policy','parallelism','dag_json','global_env','expand']
    edit_columns = add_columns

appbuilder.add_api(Pipeline_ModelView_Api)



