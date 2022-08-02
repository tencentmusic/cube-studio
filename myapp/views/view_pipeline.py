from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import re
import urllib.parse
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob
from myapp.models.model_team import Project,Project_User
from myapp.views.view_team import Project_Join_Filter
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
from myapp.views.view_task import Task_ModelView,Task_ModelView_Api
from sqlalchemy import and_, or_, select
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from myapp.project import push_message,push_admin
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
from myapp.views.view_team import filter_join_org_project
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
    json_response
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
def dag_to_pipeline(pipeline,dbsession,**kwargs):
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
    pipeline_conf = kfp.dsl.PipelineConf()


    # 设置机器选择器
    # 如果项目中设置了机器选择，就使用项目中设置的
    # node_selector = pipeline.project.get('node_selector','')
    # if not node_selector and pipeline.node_selector:
    #     node_selector = pipeline.node_selector
    #
    # if node_selector:
    #     for selector in re.split(',|;|\n|\t', str(pipeline.node_selector)):
    #         pipeline_conf.set_default_pod_node_selector(selector.split('=')[0].strip(),selector.split('=')[1].strip())

    # 渲染字符串模板变量
    def template_str(src_str):
        rtemplate = Environment(loader=BaseLoader, undefined=DebugUndefined).from_string(src_str)
        des_str = rtemplate.render(creator=pipeline.created_by.username,
                                   datetime=datetime,
                                   runner=g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
                                   uuid = uuid,
                                   pipeline_id=pipeline.id,
                                   pipeline_name=pipeline.name,
                                   cluster_name=pipeline.project.cluster['NAME'],
                                   **kwargs
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
            envs=[env.strip() for env in envs if env.strip()]
            for env in envs:
                env_key,env_value = env.split('=')[0],env.split('=')[1]
                container_envs.append(V1EnvVar(env_key,env_value))

        # 设置全局环境变量
        for global_env_key in global_envs:
            container_envs.append(V1EnvVar(global_env_key,global_envs[global_env_key]))

        # 设置task的默认环境变量
        container_envs.append(V1EnvVar("KFJ_TASK_ID", str(task.id)))
        container_envs.append(V1EnvVar("KFJ_TASK_NAME", str(task.name)))
        container_envs.append(V1EnvVar("KFJ_TASK_NODE_SELECTOR", str(task.get_node_selector())))
        container_envs.append(V1EnvVar("KFJ_TASK_VOLUME_MOUNT", str(task.volume_mount)))
        container_envs.append(V1EnvVar("KFJ_TASK_IMAGES", str(task.job_template.images)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_CPU", str(task.resource_cpu)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_MEMORY", str(task.resource_memory)))
        container_envs.append(V1EnvVar("KFJ_TASK_RESOURCE_GPU", str(task.resource_gpu.replace("+",''))))
        container_envs.append(V1EnvVar("KFJ_TASK_PROJECT_NAME", str(pipeline.project.name)))
        container_envs.append(V1EnvVar("GPU_TYPE", os.environ.get("GPU_TYPE", "NVIDIA")))
        container_envs.append(V1EnvVar("USERNAME", pipeline.created_by.username))

        container_kwargs['env']=container_envs


        # 创建工作目录
        if task.job_template.workdir and task.job_template.workdir.strip():
            container_kwargs['working_dir'] = task.job_template.workdir.strip()
        if task.working_dir and task.working_dir.strip():
            container_kwargs['working_dir'] = task.working_dir.strip()


        # # 创建label，这样能让每个pod都找到运行人。
        # container_labels={
        #     'upload-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
        #     'run-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username
        # }
        # container_kwargs['labels']=container_labels


        task_command = ''

        if task.command:
            commands = re.split('\r|\n',task.command)
            commands = [command.strip() for command in commands if command.strip()]
            if task_command:
                task_command += " && " + " && ".join(commands)
            else:
                task_command += " && ".join(commands)

        job_template_entrypoint = task.job_template.entrypoint.strip() if task.job_template.entrypoint else ''


        command=None
        if job_template_entrypoint:
            command = job_template_entrypoint

        if task_command:
            command = task_command


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
            container_kwargs['working_dir']=json.loads(task.args).get('workdir')
            ops = kfp.dsl.ContainerOp(
                name=task.name,
                image=json.loads(task.args).get('images'),
                command=['bash','-c',json.loads(task.args).get('command')],
                container_kwargs=container_kwargs,
                file_outputs = json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None
            )

        else:

            # 数组方式
            # if task_command:
            #     task_command = task_command.split(' ')
            #     task_command = [command for command in task_command if command]

            command = command.split(' ') if command else []
            command = [com for com in command if com]
            ops = kfp.dsl.ContainerOp(
                name=task.name,
                image=task.job_template.images.name,
                arguments=ops_args,
                command=command if command else None,
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
                    volumn_name = ('-'.join(temps))[:60].lower().strip('-')
                    ops=ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                           persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)))\
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=os.path.join(mount,task.pipeline.created_by.username), name=volumn_name,sub_path=task.pipeline.created_by.username))
                if "(hostpath)" in volume:
                    hostpath_name = volume.replace('(hostpath)', '').replace(' ', '')
                    temps = re.split('_|\.|/', hostpath_name)
                    temps = [temp for temp in temps if temp]
                    volumn_name = ('-'.join(temps))[:60].lower().strip('-')

                    ops = ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                             host_path=k8s_client.V1HostPathVolumeSource(path=hostpath_name))) \
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount, name=volumn_name))
                if "(configmap)" in volume:
                    configmap_name = volume.replace('(configmap)', '').replace(' ', '')
                    temps = re.split('_|\.|/', configmap_name)
                    temps = [temp for temp in temps if temp]
                    volumn_name = ('-'.join(temps))[:60].lower().strip('-')

                    ops = ops.add_volume(k8s_client.V1Volume(name=volumn_name,
                                                             config_map=k8s_client.V1ConfigMapVolumeSource(name=configmap_name))) \
                        .add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount, name=volumn_name))

                if "(memory)" in volume:
                    memory_size = volume.replace('(memory)', '').replace(' ', '').lower().replace('g','')
                    volumn_name = ('memory-%s'%memory_size)[:60].lower().strip('-')
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

        for selector in re.split(',|;|\n|\t', task.get_node_selector()):
            selector=selector.replace(' ','')
            if '=' in selector:
                ops.add_node_selector_constraint(selector.strip().split('=')[0].strip(),selector.strip().split('=')[1].strip())


        # # 根据用户身份设置机器选择器
        # if g and g.user and g.user.org:
        #     ops.add_node_selector_constraint('org-'+g.user.org,'true')
        # elif pipeline.created_by.org:  # 对定时任务，没有g存储
        #     ops.add_node_selector_constraint('org-'+pipeline.created_by.org,'true')

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
        # ops.add_pod_label('job-template', task.job_template.name)


        # 添加亲密度控制
        affinity = k8s_client.V1Affinity(
            pod_anti_affinity=k8s_client.V1PodAntiAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    k8s_client.V1WeightedPodAffinityTerm(
                        weight=80,
                        pod_affinity_term=k8s_client.V1PodAffinityTerm(
                            label_selector=k8s_client.V1LabelSelector(
                                match_labels={
                                    # 'job-template':task.job_template.name,
                                    "pipeline-id":str(pipeline.id)
                                }
                            ),
                            topology_key='kubernetes.io/hostname'
                        )
                    )
                ]
            )
        )
        ops.add_affinity(affinity)

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
            if resource_gpu and core.get_gpu(resource_gpu)[0]>0:
                ops.set_gpu_limit(core.get_gpu(resource_gpu)[0])


        all_ops[task_name]=ops

    # 这里面是真正的pipeline name  上传时指定的是version的name
    @kfp.dsl.pipeline(name=pipeline.name,description=pipeline.describe)
    def my_pipeline():
        for task_name in dag:
            get_ops(task_name)



    # pipeline运行的相关配置

    hubsecret_list = []
    for task_name in all_tasks:
        # 配置拉取秘钥。本来在contain里面，workflow在外面
        task_temp = all_tasks[task_name]
        if task_temp.job_template.images.repository.hubsecret:
            hubsecret = task_temp.job_template.images.repository.hubsecret
            if hubsecret not in hubsecret_list:
                hubsecret_list.append(hubsecret)
                pipeline_conf.image_pull_secrets.append(k8s_client.V1LocalObjectReference(name=hubsecret))


        # # 配置host 在kfp中并不生效
        # hostAliases = conf.get('HOSTALIASES', '')
        # if task_temp.job_template.hostAliases:
        #     hostAliases+="\n"+ task_temp.job_template.hostAliases
        # if hostAliases:
        #     hostAliases_list = re.split('\r|\n', hostAliases)
        #     hostAliases_list = [host.strip() for host in hostAliases_list if host.strip()]
        #     for row in hostAliases_list:
        #         hosts = row.strip().split(' ')
        #         hosts = [host for host in hosts if host]
        #
        #         if len(hosts) > 1:
        #             pipeline_conf.set_host_aliases(ip=hosts[0],hostnames=hosts[1:])


    # 配置默认拉取策略
    # if pipeline.image_pull_policy:
    pipeline_conf.image_pull_policy = conf.get('IMAGE_PULL_POLICY','Always')

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
def upload_pipeline(pipeline_file,pipeline_name,kfp_host,pipeline_argo_id):

    if not pipeline_file:
        return None,None

    file = open(pipeline_name + '.yaml', mode='wb')
    file.write(bytes(pipeline_file,encoding='utf-8'))
    file.close()
    client = kfp.Client(kfp_host)   # pipeline.project.cluster.get('KFP_HOST')
    pipeline_argo = None
    if pipeline_argo_id:
        try:
            pipeline_argo = client.get_pipeline(pipeline_argo_id)
        except Exception as e:
            logging.error(e)

    if pipeline_argo:
        pipeline_argo_version = client.upload_pipeline_version(pipeline_package_path=pipeline_name + '.yaml', pipeline_version_name=pipeline_name+"_version_at_"+datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),pipeline_id=pipeline_argo.id)
        time.sleep(1)   # 因为创建是异步的，要等k8s反应，所以有时延
        return pipeline_argo.id,pipeline_argo_version.id
    else:
        exist_pipeline_argo_id = None
        try:
            exist_pipeline_argo_id = client.get_pipeline_id(pipeline_name)
        except Exception as e:
            logging.error(e)

        if exist_pipeline_argo_id:
            pipeline_argo_version = client.upload_pipeline_version(pipeline_package_path=pipeline_name + '.yaml',pipeline_version_name=pipeline_name + "_version_at_" + datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),pipeline_id=exist_pipeline_argo_id)
            time.sleep(1)
            return exist_pipeline_argo_id,pipeline_argo_version.id
        else:
            pipeline_argo = client.upload_pipeline(pipeline_name + '.yaml', pipeline_name=pipeline_name)
            time.sleep(1)
            return pipeline_argo.id,pipeline_argo.default_version.id



# @pysnooper.snoop(watch_explode=())
def run_pipeline(pipeline_file,pipeline_name,kfp_host,pipeline_argo_id,pipeline_argo_version_id):
    # logging.info(pipeline)
    # return
    # 如果没值就先upload
    if not pipeline_argo_id or not pipeline_argo_version_id:
        pipeline_argo_id,pipeline_argo_version_id = upload_pipeline(pipeline_file,pipeline_name,kfp_host,pipeline_argo_id)   # 必须上传新版本

    client = kfp.Client(kfp_host)
    # 先创建一个实验，在在这个实验中运行指定pipeline
    experiment=None
    try:
        experiment = client.get_experiment(experiment_name=pipeline_name)
    except Exception as e:
        logging.error(e)
    if not experiment:
        try:
            experiment = client.create_experiment(name=pipeline_name,description=pipeline_name)  # 现在要求describe不能是中文了
        except Exception as e:
            print(e)
            return None,None,None
    # 直接使用pipeline最新的版本运行
    try:
        run = client.run_pipeline(experiment_id = experiment.id,pipeline_id=pipeline_argo_id,version_id=pipeline_argo_version_id,job_name=pipeline_name+"_version_at_"+datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
        return pipeline_argo_id, pipeline_argo_version_id, run.id
    except Exception as e:
        print(e)
        raise e





class Pipeline_ModelView_Base():
    label_title='任务流'
    datamodel = SQLAInterface(Pipeline)
    check_redirect_list_url = conf.get('MODEL_URLS',{}).get('pipeline','')

    base_permissions = ['can_show','can_edit','can_list','can_delete','can_add']
    base_order = ("changed_on", "desc")
    # order_columns = ['id','changed_on']
    order_columns = ['id']

    list_columns = ['id','project','pipeline_url','creator','modified']
    cols_width={
        "id":{"type": "ellip2", "width": 100},
        "project": {"type": "ellip2", "width": 200},
        "pipeline_url":{"type": "ellip2", "width": 500},
        "modified": {"type": "ellip2", "width": 150}
    }
    add_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','expired_limit','parallelism','global_env','alert_status','alert_user','parameter']
    show_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','expired_limit','parallelism','global_env','dag_json','pipeline_file','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand','parameter']
    # show_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    search_columns = ['id', 'created_by', 'name', 'describe', 'schedule_type', 'project']
    edit_columns = add_columns


    base_filters = [["id", Pipeline_Filter, lambda: []]]  # 设置权限过滤器
    conv = GeneralModelConverter(datamodel)


    add_form_extra_fields = {

        "name": StringField(
            _(datamodel.obj.lab('name')),
            description="英文名(字母、数字、- 组成)，最长50个字符",
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9\-]*[a-z0-9]$"),Length(1,54),DataRequired()]
        ),
        "describe": StringField(
            _(datamodel.obj.lab('describe')),
            description="中文描述",
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "project":QuerySelectField(
            _(datamodel.obj.lab('project')),
            query_factory=filter_join_org_project,
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

        "depends_on_past": BooleanField(
            _(datamodel.obj.lab('depends_on_past')),
            description="任务运行是否依赖上一次的示例状态",
            default=True
        ),
        "max_active_runs": IntegerField(
            _(datamodel.obj.lab('max_active_runs')),
            description="当前pipeline可同时运行的任务流实例数目",
            widget=BS3TextFieldWidget(),
            default=1,
            validators=[DataRequired()]
        ),
        "expired_limit": IntegerField(
            _(datamodel.obj.lab('expired_limit')),
            description="定时调度最新实例限制数目，0表示不限制",
            widget=BS3TextFieldWidget(),
            default=1,
            validators=[DataRequired()]
        ),
        "parallelism": IntegerField(
            _(datamodel.obj.lab('parallelism')),
            description="一个任务流实例中可同时运行的task数目",
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
            default='once',
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


    # 检测是否具有编辑权限，只有creator和admin可以编辑
    def check_edit_permission(self, item):
        user_roles = [role.name.lower() for role in list(get_user_roles())]
        if "admin" in user_roles:
            return True
        if g.user and g.user.username and hasattr(item,'created_by'):
            if g.user.username==item.created_by.username:
                return True
        flash('just creator can edit/delete ', 'warning')
        return False


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
        item.name = item.name.replace('_', '-')[0:54].lower().strip('-')
        # item.alert_status = ','.join(item.alert_status)
        self.pipeline_args_check(item)
        item.create_datetime=datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()
        item.parameter = json.dumps({"cronjob_start_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, indent=4, ensure_ascii=False)



    # @pysnooper.snoop()
    def pre_update(self, item):

        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        else:
            item.expand='{}'
        item.name = item.name.replace('_', '-')[0:54].lower()
        # item.alert_status = ','.join(item.alert_status)
        self.merge_upstream(item)
        self.pipeline_args_check(item)
        item.change_datetime = datetime.datetime.now()
        if item.parameter:
            item.parameter = json.dumps(json.loads(item.parameter),indent=4,ensure_ascii=False)
        else:
            item.parameter = '{}'

        if (item.schedule_type=='crontab' and self.src_item_json.get("schedule_type")=='once') or (item.cron_time!=self.src_item_json.get("cron_time",'')):
            parameter = json.loads(item.parameter if item.parameter else '{}')
            parameter.update({"cronjob_start_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
            item.parameter = json.dumps(parameter,indent=4,ensure_ascii=False)

        # 限制提醒
        if item.schedule_type=='crontab':
            if not item.project.node_selector:
                flash('无法保障公共集群的稳定性，定时任务请选择专门的日更集群项目组','warning')
            else:
                org = item.project.node_selector.replace('org=','')
                if not org or org=='public':
                    flash('无法保障公共集群的稳定性，定时任务请选择专门的日更集群项目组','warning')


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
        if "(废弃)" not in pipeline.describe:
            pipeline.describe+="(废弃)"
        pipeline.schedule_type='once'
        pipeline.expand=""
        pipeline.dag_json="{}"
        db.session.commit()


    @expose("/my/list/")
    def my(self):
        try:
            user_id=g.user.id
            if user_id:
                pipelines = db.session.query(Pipeline).filter_by(created_by_fk=user_id).all()
                back=[]
                for pipeline in pipelines:
                    back.append(pipeline.to_json())
                return json_response(message='success',status=0,result=back)
        except Exception as e:
            print(e)
            return json_response(message=str(e),status=-1,result={})


    @expose("/demo/list/")
    def demo(self):
        try:
            pipelines = db.session.query(Pipeline).filter(Pipeline.parameter.contains('"demo": "true"')).all()
            back=[]
            for pipeline in pipelines:
                back.append(pipeline.to_json())
            return json_response(message='success',status=0,result=back)
        except Exception as e:
            print(e)
            return json_response(message=str(e),status=-1,result={})


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

    



    # 删除手动发起的workflow，不删除定时任务发起的workflow
    def delete_bind_crd(self,crds):

        for crd in crds:
            try:
                run_id = json.loads(crd['labels']).get("run-id",'')
                if run_id:
                    # 定时任务发起的不能清理
                    run_history = db.session.query(RunHistory).filter_by(run_id=run_id).first()
                    if run_history:
                        continue

                    db_crd = db.session.query(Workflow).filter_by(name=crd['name']).first()
                    pipeline = db_crd.pipeline
                    if pipeline:
                        k8s_client = py_k8s.K8s(pipeline.project.cluster.get('KUBECONFIG',''))
                    else:
                        k8s_client = py_k8s.K8s()

                    k8s_client.delete_workflow(
                        all_crd_info = conf.get("CRD_INFO", {}),
                        namespace=crd['namespace'],
                        run_id = run_id
                    )
                    # push_message(conf.get('ADMIN_USER', '').split(','),'%s手动运行新的pipeline %s，进而删除旧的pipeline run-id: %s' % (pipeline.created_by.username,pipeline.describe,run_id,))
                    if db_crd:
                        db_crd.status='Deleted'
                        db_crd.change_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

        pipeline.delete_old_task()

        time.sleep(1)
        back_crds = pipeline.get_workflow()

        # 把消息加入到源数据库
        for crd in back_crds:
            try:
                workflow = db.session.query(Workflow).filter_by(name=crd['name']).first()
                if not workflow:
                    username = ''
                    labels = json.loads(crd['labels'])
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
        # not_running_crds = back_crds  # [crd for crd in back_crds if 'running' not in crd['status'].lower()]
        self.delete_bind_crd(back_crds)

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
        pipeline_argo_id, version_id, run_id = run_pipeline(
            pipeline_file=pipeline.pipeline_file,
            pipeline_name=pipeline.name,
            kfp_host=pipeline.project.cluster.get('KFP_HOST'),
            pipeline_argo_id=pipeline.pipeline_argo_id,
            pipeline_argo_version_id=pipeline.version_id
        )   # 会根据版本号是否为空决定是否上传
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

        pipeline.dag_json = pipeline.fix_dag_json()  # 修正 dag_json
        pipeline.expand = json.dumps(pipeline.fix_expand(), indent=4, ensure_ascii=False)   # 修正 前端expand字段缺失
        pipeline.expand = json.dumps(pipeline.fix_position(), indent=4, ensure_ascii=False)  # 修正 节点中心位置到视图中间

        # # 自动排版
        # db_tasks = pipeline.get_tasks(db.session)
        # if db_tasks:
        #     try:
        #         tasks={}
        #         for task in db_tasks:
        #             tasks[task.name]=task.to_json()
        #         expand = core.fix_task_position(pipeline.to_json(),tasks,json.loads(pipeline.expand))
        #         pipeline.expand=json.dumps(expand,indent=4,ensure_ascii=False)
        #         db.session.commit()
        #     except Exception as e:
        #         print(e)


        db.session.commit()
        print(pipeline_id)
        url = '/static/appbuilder/vison/index.html?pipeline_id=%s'%pipeline_id  # 前后端集成完毕，这里需要修改掉
        data = {
            "url": url
        }
        return redirect('/frontend/showOutLink?url=%s'%urllib.parse.quote(url, safe=""))
        # 返回模板
        # return self.render_template('link.html', data=data)


    # # @event_logger.log_this
    @expose("/web/log/<pipeline_id>", methods=["GET"])
    def web_log(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        if pipeline.run_id:
            data = {
                "url": pipeline.project.cluster.get('PIPELINE_URL') + "runs/details/" + pipeline.run_id,
                "target": "div.page_f1flacxk:nth-of-type(0)",   # "div.page_f1flacxk:nth-of-type(0)",
                "delay":500,
                "loading": True
            }
            # 返回模板
            if pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
                return self.render_template('link.html', data=data)
            else:
                return self.render_template('external_link.html', data=data)
        else:
            flash('no running instance','warning')
            return redirect('/pipeline_modelview/web/%s'%pipeline.id)


    # # @event_logger.log_this
    @expose("/web/monitoring/<pipeline_id>", methods=["GET"])
    def web_monitoring(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()
        if pipeline.run_id:
            url = pipeline.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_TASK_PATH')+ pipeline.name
            return redirect(url)
            # data = {
            #     "url": pipeline.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_TASK_PATH') + pipeline.name,
            #     # "target": "div.page_f1flacxk:nth-of-type(0)",   # "div.page_f1flacxk:nth-of-type(0)",
            #     "delay":1000,
            #     "loading": True
            # }
            # # 返回模板
            # if pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
            #     return self.render_template('link.html', data=data)
            # else:
            #     return self.render_template('external_link.html', data=data)
        else:
            flash('no running instance','warning')
            return redirect('/pipeline_modelview/web/%s'%pipeline.id)

    # # @event_logger.log_this
    @expose("/web/pod/<pipeline_id>", methods=["GET"])
    def web_pod(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        data = {
            "url": pipeline.project.cluster.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (conf.get('PIPELINE_NAMESPACE'), pipeline.name.replace('_', '-').lower()),
            "target":"div.kd-chrome-container.kd-bg-background",
            "delay":500,
            "loading": True
        }
        # 返回模板
        if pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)


    # @pysnooper.snoop(watch_explode=('expand'))
    def copy_db(self,pipeline):
        new_pipeline = pipeline.clone()
        expand = json.loads(pipeline.expand) if pipeline.expand else {}
        new_pipeline.name = new_pipeline.name.replace('_', '-') + "-" + uuid.uuid4().hex[:4]
        new_pipeline.created_on = datetime.datetime.now()
        new_pipeline.changed_on = datetime.datetime.now()
        db.session.add(new_pipeline)
        db.session.commit()

        def change_node(src_task_id, des_task_id):
            for node in expand:
                if 'source' not in node:
                    # 位置信息换成新task的id
                    if int(node['id']) == int(src_task_id):
                        node['id'] = str(des_task_id)
                else:
                    if int(node['source']) == int(src_task_id):
                        node['source'] = str(des_task_id)
                    if int(node['target']) == int(src_task_id):
                        node['target'] = str(des_task_id)

        # 复制绑定的task，并绑定新的pipeline
        for task in pipeline.get_tasks():
            new_task = task.clone()
            new_task.pipeline_id = new_pipeline.id
            new_task.create_datetime = datetime.datetime.now()
            new_task.change_datetime = datetime.datetime.now()
            db.session.add(new_task)
            db.session.commit()
            change_node(task.id, new_task.id)

        new_pipeline.expand = json.dumps(expand)
        db.session.commit()
        return new_pipeline

    # # @event_logger.log_this
    @expose("/copy_pipeline/<pipeline_id>", methods=["GET", "POST"])
    def copy_pipeline(self,pipeline_id):
        print(pipeline_id)
        message=''
        try:
            pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
            new_pipeline = self.copy_db(pipeline)
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
                self.copy_db(pipeline)
        except InvalidRequestError:
            db.session.rollback()
        except Exception as e:
            logging.error(e)
            raise e

        return redirect(request.referrer)



    @expose("/config/<pipeline_id>",methods=("GET",'POST'))
    # @pysnooper.snoop()
    def pipeline_config(self,pipeline_id):
        print(pipeline_id)
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()

        if not pipeline:
            return jsonify({
                "status":1,
                "message":"任务流不存在",
                "result":{}
            })
        if request.method.lower()=='post':

            if g.user.username != pipeline.created_by.username and not g.user.is_admin():
                return jsonify({
                    "result": {},
                    "message": "只有创建者或管理员可修改",
                    "status": -1
                })


        config = {
            "id":pipeline.id,
            "name":pipeline.name,
            "label":pipeline.describe,
            "project":pipeline.project.describe,
            "pipeline_jump_button": [
                {
                    "name":"调度实例",
                    "action_url":conf.get('MODEL_URLS',{}).get('workflow')+'?filter='+urllib.parse.quote(json.dumps([{"key":"label","value":'"pipeline-id": "%s"'%pipeline_id}],ensure_ascii=False)),
                    "icon_svg":'<svg t="1658745802720" class="icon" viewBox="0 0 1239 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="3024" width="200" height="200"><path d="M662.929155 1023.027623a83.489741 83.489741 0 0 1-33.395896-7.235778 86.272732 86.272732 0 0 1-51.207041-79.593553v-311.695031L140.839976 1001.876889a82.376544 82.376544 0 0 1-89.612321 15.028153A86.272732 86.272732 0 0 1 0.020614 936.198292V86.82933A86.272732 86.272732 0 0 1 51.784253 7.235778a82.376544 82.376544 0 0 1 89.612322 15.584751l436.373044 377.373628V86.82933a86.272732 86.272732 0 0 1 51.76364-79.593552 82.376544 82.376544 0 0 1 90.16892 15.584751L1210.621854 445.278617a87.385929 87.385929 0 0 1 0 128.574201L718.588982 1001.876889a82.933142 82.933142 0 0 1-55.659827 21.150734z m-578.862202-87.385929l492.58947-424.127883-490.919675-426.910874v851.595355z m577.749005 0l492.589471-424.127883-490.919676-426.910874v851.595355z" fill="#245fd1" p-id="3025"></path></svg>'
                },
                {
                    "name": "日志",
                    "action_url": '/pipeline_modelview/web/log/%s'%pipeline_id,
                    "icon_svg": '<svg t="1658745990874" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2268" width="200" height="200"><path d="M819.9 472.9L675 723.9l1.7 99.4 86.9-48.3 144.9-251-88.6-51.1zM871.1 384.3L837 443.4l88.6 51.1 34.1-59.1-88.6-51.1zM544.3 703h-288c-17.7 0-32 14.3-32 32s14.3 32 32 32h288c17.7 0 32-14.3 32-32s-14.3-32-32-32zM256.3 511c-17.7 0-32 14.3-32 32s14.3 32 32 32h384c17.7 0 32-14.3 32-32s-14.3-32-32-32h-384zM256.3 319c-17.7 0-32 14.3-32 32s14.3 32 32 32h384c17.7 0 32-14.3 32-32s-14.3-32-32-32h-384zM288 64h64v160h-64zM384 128h128v64H384zM544 64h64v160h-64z" p-id="2269" fill="#245fd1"></path><path d="M768 864c0 17.7-14.3 32-32 32H160c-17.7 0-32-14.3-32-32V224c0-17.7 14.3-32 32-32h96v-64h-96c-53 0-96 43-96 96v640c0 53 43 96 96 96h576c53 0 96-43 96-96V686.7L768 798v66zM832 224c0-53-43-96-96-96h-96v64h96c17.7 0 32 14.3 32 32v311l64-111.3V224z" p-id="2270" fill="#245fd1"></path></svg>'
                },
                {
                    "name": "容器",
                    "action_url": '/pipeline_modelview/web/pod/%s'%pipeline_id,
                    "icon_svg": '<svg t="1658746095720" class="icon" viewBox="0 0 1472 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4861" width="200" height="200"><path d="M1449.66628 358.737a233.848 233.848 0 0 0-166.348-35.445 268.717 268.717 0 0 0-108.127-152.273l-31.158-20.026-22.265 30.455a258.736 258.736 0 0 0-22.01 265.39 177.353 177.353 0 0 1-74.28 21.241h-24.953V309.536H830.08228V0H624.44928v154.768H287.27328v154.704H118.68528V468.08H8.44728L3.26528 504.42a493.032 493.032 0 0 0 95.97 353.3c90.149 110.11 234.232 165.964 428.284 165.964a749.848 749.848 0 0 0 585.42-255.025 804.871 804.871 0 0 0 139.86-226.874c187.718-3.391 213.246-134.359 214.27-139.99l4.863-27.447-22.01-15.61z m-766.291-49.84v-92.068h87.717v92.068h-87.717z m-337.176 154.64v-92.068h87.59v92.068h-87.59z m168.588 0v-92.068h87.589v92.068h-87.589z m168.588 0v-92.068h87.717v92.068h-87.717z m170.38-92.068h87.524v92.068h-87.525v-92.068zM683.37428 62.125h87.717v92.003h-87.717V62.125zM514.78728 216.829h87.589v92.068h-87.525v-92.068z m-168.588 0h87.59v92.068h-87.59v-92.068zM177.61228 371.47h87.525v92.068H177.61228v-92.068zM527.19928 938.4a609.348 609.348 0 0 1-235-40.564 399.493 399.493 0 0 0 151.058-66.092 44.018 44.018 0 0 0 7.87-57.582 39.54 39.54 0 0 0-54.575-11.9 375.18 375.18 0 0 1-215.998 62.508 262.639 262.639 0 0 1-19.194-21.433 392.455 392.455 0 0 1-79.591-249.523h943.9a250.035 250.035 0 0 0 155.216-62.06l4.99-4.671a682.157 682.157 0 0 1-658.42 451.636z m699.432-482.412l-25.144-1.215-15.163-21.178a186.566 186.566 0 0 1-21.626-161.358 145.619 145.619 0 0 1 42.483 100.769l-1.663 60.525 54.83-18.682a205.505 205.505 0 0 1 111.07-1.664 170.123 170.123 0 0 1-144.787 42.803zM544.41028 629.31a69.738 69.738 0 1 1-66.412 69.674 68.139 68.139 0 0 1 66.412-69.674z m0 85.413a15.74 15.74 0 1 0-14.971-15.675 15.291 15.291 0 0 0 14.97 15.675z m0 0" p-id="4862" fill="#245fd1"></path></svg>'
                },
                {
                    "name": "定时记录",
                    "action_url": conf.get('MODEL_URLS', {}).get('runhistory')+'?filter='+urllib.parse.quote(json.dumps([{"key":"pipeline","value":pipeline_id}],ensure_ascii=False)),
                    "icon_svg": '<svg t="1658746184808" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7122" width="200" height="200"><path d="M972.1 345.3c-7.6 0-14.8-4.3-18.2-11.6-58-126.1-167.3-221.3-299.9-261.4-10.6-3.2-16.6-14.4-13.4-24.9S655 30.8 665.5 34C809 77.4 927.3 180.5 990.1 317c4.6 10 0.2 21.9-9.8 26.5-2.5 1.2-5.4 1.8-8.2 1.8zM55.3 345.2c-2.8 0-5.6-0.6-8.4-1.8-10-4.6-14.4-16.5-9.8-26.5C98.2 184.2 212.8 82.3 351.4 37.2c10.5-3.4 21.8 2.3 25.2 12.8 3.4 10.5-2.3 21.8-12.8 25.2C235.8 116.8 130 211 73.5 333.5c-3.4 7.4-10.7 11.7-18.2 11.7zM510.3 181.4c47.9 0 94.3 9.4 138 27.9 42.2 17.9 80.2 43.4 112.7 76 32.6 32.6 58.2 70.5 76 112.7 18.5 43.7 27.9 90.1 27.9 138 0 47.9-9.4 94.3-27.9 138-17.9 42.2-43.4 80.2-76 112.7-32.6 32.6-70.5 58.2-112.7 76-43.7 18.5-90.1 27.9-138 27.9s-94.3-9.4-138-27.9c-42.2-17.9-80.2-43.4-112.7-76-32.6-32.6-58.2-70.5-76-112.7-18.5-43.7-27.9-90.1-27.9-138 0-47.9 9.4-94.3 27.9-138 17.9-42.2 43.4-80.2 76-112.7s70.5-58.2 112.7-76c43.7-18.5 90.1-27.9 138-27.9m0-41.8c-218.9 0-396.4 177.5-396.4 396.4s177.5 396.4 396.4 396.4S906.7 755 906.7 536.1 729.3 139.6 510.3 139.6z" fill="#245fd1" p-id="7123"></path><path d="M513.7 556.1c-3.8 0-7.5-1.1-10.8-3.2-5.7-3.7-9.2-10-9.2-16.8V232.5c0-11 9-20 20-20s20 9 20 20V505l171-77.9c10-4.6 21.9-0.1 26.5 9.9 4.6 10.1 0.1 21.9-9.9 26.5L522 554.3c-2.7 1.2-5.5 1.8-8.3 1.8zM202 991.8c-5.1 0-10.2-2-14.1-5.9-7.8-7.8-7.8-20.5 0-28.3l99.8-99.8c7.8-7.8 20.5-7.8 28.3 0s7.8 20.5 0 28.3l-99.8 99.8c-3.9 3.9-9 5.9-14.2 5.9zM818.6 991.8c-5.1 0-10.2-2-14.1-5.9l-99.8-99.8c-7.8-7.8-7.8-20.5 0-28.3 7.8-7.8 20.5-7.8 28.3 0l99.8 99.8c7.8 7.8 7.8 20.5 0 28.3-4 3.9-9.1 5.9-14.2 5.9z" fill="#245fd1" p-id="7124"></path></svg>'
                },
                {
                    "name": "监控",
                    "action_url": '/pipeline_modelview/web/monitoring/%s'%pipeline_id,
                    "icon_svg": '<svg t="1658746229697" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="8210" width="200" height="200"><path d="M512 302.037c-83.338 0-161.688 32.454-220.617 91.383C232.454 452.348 200 530.698 200 614.036c0 17.673 14.327 32 32 32s32-14.327 32-32c0-66.243 25.796-128.521 72.638-175.362 46.841-46.841 109.119-72.638 175.362-72.638s128.521 25.796 175.362 72.638C734.203 485.515 760 547.793 760 614.036c0 17.673 14.327 32 32 32s32-14.327 32-32c0-83.338-32.454-161.688-91.383-220.617C673.688 334.49 595.338 302.037 512 302.037z" fill="#245fd1" p-id="8211"></path><path d="M512 158C264.576 158 64 358.576 64 606c0 89.999 26.545 173.796 72.224 244h751.553C933.455 779.796 960 695.999 960 606c0-247.424-200.576-448-448-448z m339.288 628H172.712C143.373 730.813 128 669.228 128 606c0-51.868 10.144-102.15 30.15-149.451 19.337-45.719 47.034-86.792 82.321-122.078 35.286-35.287 76.359-62.983 122.078-82.321C409.85 232.144 460.132 222 512 222c51.868 0 102.15 10.144 149.451 30.15 45.719 19.337 86.792 47.034 122.078 82.321 35.287 35.286 62.983 76.359 82.321 122.078C885.856 503.85 896 554.132 896 606c0 63.228-15.373 124.813-44.712 180z" fill="#245fd1" p-id="8212"></path><path d="M532.241 586.466a79.753 79.753 0 0 0-29.217 5.529l-69.087-69.087c-12.497-12.499-32.758-12.497-45.255-0.001-12.497 12.497-12.497 32.759 0 45.255l69.088 69.088a79.753 79.753 0 0 0-5.529 29.217c0 44.183 35.817 80 80 80s80-35.817 80-80-35.818-80.001-80-80.001z" fill="#245fd1" p-id="8213"></path></svg>'
                }
            ],
            "pipeline_run_button": [
                {
                    "name": "运行",
                    "action_url": "/pipeline_modelview/run_pipeline/%s"%pipeline.id
                },
                {
                    "name": "复制",
                    "action_url": "/pipeline_modelview/copy_pipeline/%s" % pipeline.id
                }
            ],
            "message": "success",
            "status": 0
        }


        return jsonify(config)



class Pipeline_ModelView(Pipeline_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Pipeline)
    # base_order = ("changed_on", "desc")
    # order_columns = ['changed_on']

appbuilder.add_view(Pipeline_ModelView,"任务流",href="/pipeline_modelview/list/?_flt_0_created_by=",icon = 'fa-sitemap',category = '训练')


# 添加api
class Pipeline_ModelView_Api(Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Pipeline)
    route_base = '/pipeline_modelview/api'
    # show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    list_columns = ['id','project','pipeline_url','creator','modified']
    add_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','global_env','alert_status','expand']
    edit_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','dag_json','global_env','alert_status','expand','created_by']

    related_views = [Task_ModelView_Api,]

    def pre_add_get(self):
        self.default_filter = {
            "created_by": g.user.id
        }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields


appbuilder.add_api(Pipeline_ModelView_Api)




