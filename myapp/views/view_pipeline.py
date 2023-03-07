from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import urllib.parse
from sqlalchemy.exc import InvalidRequestError

from myapp.models.model_job import Task,Pipeline,Workflow, RunHistory
from myapp.models.model_team import Project
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import jsonify
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from jinja2 import Environment, BaseLoader, DebugUndefined
import os
from wtforms.validators import DataRequired, Length, Regexp
from myapp.views.view_task import Task_ModelView,Task_ModelView_Api
from sqlalchemy import or_
from myapp.exceptions import MyappException
from wtforms import BooleanField, IntegerField,StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,Select2ManyWidget,Select2Widget,BS3TextAreaFieldWidget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
import re,copy
from kubernetes.client.models import (
    V1EnvVar,V1SecurityContext
)
from .baseApi import (
    MyappModelRestApi
)
from flask import (
    flash,
    g,
    make_response,
    redirect,
    request
)
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project
import pysnooper
from kubernetes import client
from .base import (
    DeleteMixin,
    get_user_roles,
    MyappFilter,
    MyappModelView,
    json_response
)

from flask_appbuilder import expose
import datetime,time,json
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



def make_workflow_yaml(pipeline,workflow_label,hubsecret_list,dag_templates,containers_templates):
    name = pipeline.name+"-"+uuid.uuid4().hex[:4]
    workflow_label['workflow-name']=name
    workflow_crd_json={
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            # "generateName": pipeline.name+"-",
            "annotations": {
                "name": pipeline.name,
                "description":pipeline.describe.encode("unicode_escape").decode('utf-8')
            },
            "name":name,
            "labels": workflow_label,
            "namespace":json.loads(pipeline.project.expand).get('PIPELINE_NAMESPACE', conf.get('PIPELINE_NAMESPACE'))
        },
        "spec": {
            "ttlStrategy":{
                "secondsAfterCompletion":10800,  # 3个小时候自动删除
                "ttlSecondsAfterFinished":10800, # 3个小时候自动删除
            },
            "archiveLogs": True,  # 打包日志
            "entrypoint": pipeline.name,
            "templates": [
                {
                    "name": pipeline.name,
                    "dag": {
                        "tasks": dag_templates
                    }
                }
            ]+containers_templates,
            "arguments": {
                "parameters": []
            },
            "serviceAccountName": "pipeline-runner",
            "parallelism": int(pipeline.parallelism),
            "imagePullSecrets": [
                {
                    "name": hubsecret
                } for hubsecret in hubsecret_list
            ]
        }
    }
    return workflow_crd_json

# 转化为worfklow的yaml
# @pysnooper.snoop()
def dag_to_pipeline(pipeline,dbsession,workflow_label=None,**kwargs):

    pipeline.dag_json = pipeline.fix_dag_json(dbsession)
    dbsession.commit()
    dag = json.loads(pipeline.dag_json)

    # 如果dag为空，就直接退出
    if not dag:
        return None,None

    all_tasks = {}
    for task_name in dag:
        # 使用临时连接，避免连接中断的问题
        # try:
            # db.session().ping()
            task = dbsession.query(Task).filter_by(name=task_name, pipeline_id=pipeline.id).first()
            if not task:
                raise MyappException('task %s not exist ' % task_name)
            all_tasks[task_name]=task

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


    def make_dag_template():
        dag_template=[]
        for task_name in dag:
            dag_template.append({
                "name": task_name,
                "template": task_name,
                "dependencies":dag[task_name].get('upstream',[])
            })
        return dag_template

    # @pysnooper.snoop()
    def make_container_template(task_name):
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


        # 设置环境变量
        container_envs = []
        if task.job_template.env:
            envs = re.split('\r|\n',task.job_template.env)
            envs=[env.strip() for env in envs if env.strip()]
            for env in envs:
                env_key,env_value = env.split('=')[0],env.split('=')[1]
                container_envs.append((env_key,env_value))

        # 设置全局环境变量
        for global_env_key in global_envs:
            container_envs.append((global_env_key,global_envs[global_env_key]))

        # 设置task的默认环境变量
        container_envs.append(("KFJ_TASK_ID", str(task.id)))
        container_envs.append(("KFJ_TASK_NAME", str(task.name)))
        container_envs.append(("KFJ_TASK_NODE_SELECTOR", str(task.get_node_selector())))
        container_envs.append(("KFJ_TASK_VOLUME_MOUNT", str(task.volume_mount)))
        container_envs.append(("KFJ_TASK_IMAGES", str(task.job_template.images)))
        container_envs.append(("KFJ_TASK_RESOURCE_CPU", str(task.resource_cpu)))
        container_envs.append(("KFJ_TASK_RESOURCE_MEMORY", str(task.resource_memory)))
        container_envs.append(("KFJ_TASK_RESOURCE_GPU", str(task.resource_gpu.replace("+",''))))
        container_envs.append(("KFJ_TASK_PROJECT_NAME", str(pipeline.project.name)))
        container_envs.append(("GPU_TYPE", os.environ.get("GPU_TYPE", "NVIDIA")))
        container_envs.append(("USERNAME", pipeline.created_by.username))


        # 创建工作目录
        working_dir = None
        if task.job_template.workdir and task.job_template.workdir.strip():
            working_dir = task.job_template.workdir.strip()
        if task.working_dir and task.working_dir.strip():
            working_dir = task.working_dir.strip()

        # 配置启动命令
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

        images=task.job_template.images.name
        command = command.split(' ') if command else []
        command = [com for com in command if com]
        arguments = ops_args
        file_outputs = json.loads(task.outputs) if task.outputs and json.loads(task.outputs) else None

        if task.job_template.name==conf.get('CUSTOMIZE_JOB'):
            working_dir=json.loads(task.args).get('workdir')
            images = json.loads(task.args).get('images')
            command = ['bash', '-c', json.loads(task.args).get('command')]
            arguments = []


        # 添加用户自定义挂载
        k8s_volumes = []
        k8s_volume_mounts = []
        task.volume_mount=task.volume_mount.strip() if task.volume_mount else ''
        if task.volume_mount:
            try:
                k8s_volumes,k8s_volume_mounts = py_k8s.K8s.get_volume_mounts(task.volume_mount,pipeline.created_by.username)
            except Exception as e:
                print(e)


        # 添加node selector
        node_selector={}
        for selector in re.split(',|;|\n|\t', task.get_node_selector()):
            selector=selector.replace(' ','')
            if '=' in selector:
                node_selector[selector.strip().split('=')[0].strip()] = selector.strip().split('=')[1].strip()

        # 添加pod label
        pod_label={
            "pipeline-id": str(pipeline.id),
            "pipeline-name": str(pipeline.name),
            "task-name": str(task.name),
            "run-id": global_envs.get('KFJ_RUN_ID', ''),
            "task-id": str(task.id),
            'upload-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
            'run-rtx': g.user.username if g and g.user and g.user.username else pipeline.created_by.username,
            'pipeline-rtx': pipeline.created_by.username
        }
        pod_annotations={
            "task-label": str(task.label).encode("unicode_escape").decode('utf-8'),
            'job-template': task.job_template.describe.encode("unicode_escape").decode('utf-8')
        }

        # 设置资源限制
        resource_cpu = task.job_template.get_env('TASK_RESOURCE_CPU') if task.job_template.get_env('TASK_RESOURCE_CPU') else task.resource_cpu
        resource_gpu = task.job_template.get_env('TASK_RESOURCE_GPU') if task.job_template.get_env('TASK_RESOURCE_GPU') else task.resource_gpu
        resource_memory = task.job_template.get_env('TASK_RESOURCE_MEMORY') if task.job_template.get_env('TASK_RESOURCE_MEMORY') else task.resource_memory

        resources_requests = resources_limits = {}


        if resource_memory:
            if not '~' in resource_memory:
                resources_requests['memory']=resource_memory
                resources_limits['memory'] = resource_memory
            else:
                resources_requests['memory'] = resource_memory.split("~")[0]
                resources_limits['memory'] = resource_memory.split("~")[1]

        if resource_cpu:
            if not '~' in resource_cpu:
                resources_requests['cpu'] = resource_cpu
                resources_limits['cpu'] = resource_cpu

            else:
                resources_requests['cpu'] = resource_cpu.split("~")[0]
                resources_limits['cpu'] = resource_cpu.split("~")[1]

        if resource_gpu:
            if resource_gpu and int(core.get_gpu(resource_gpu)[0])>0:
                gpu_drive_type = conf.get("GPU_DRIVE_TYPE", "NVIDIA")
                gpu_num = int(core.get_gpu(resource_gpu)[0])
                if gpu_drive_type == 'NVIDIA':
                    resources_requests['nvidia.com/gpu'] = str(gpu_num)
                    resources_limits['nvidia.com/gpu'] = str(gpu_num)

        # 配置host
        hostAliases={}

        global_hostAliases = conf.get('HOSTALIASES', '')
        global_hostAliases=''
        if task_temp.job_template.hostAliases:
            global_hostAliases+="\n"+ task_temp.job_template.hostAliases
        if global_hostAliases:
            hostAliases_list = re.split('\r|\n', global_hostAliases)
            hostAliases_list = [host.strip() for host in hostAliases_list if host.strip()]
            for row in hostAliases_list:
                hosts = row.strip().split(' ')
                hosts = [host for host in hosts if host]
                if len(hosts) > 1:
                    hostAliases[hosts[1]]=hosts[0]


        if task.skip:
            command = ["echo", "skip"]
            arguments=None
            resources_requests=None
            resources_limits=None

        # print(command)
        task_template = {
            "name": task.name,  # 因为同一个
            "outputs":{
                "artifacts":[
                    {
                        "name": "metric",
                        "path": "/metric.json",
                        "optional":True
                    }
                ]
            },
            "container": {
                "name":task.name + "-" + uuid.uuid4().hex[:4],
                "command": command,
                "args":arguments,
                "env": [
                    {
                        "name": item[0],
                        "value": item[1]
                    } for item in container_envs
                ],
                "image": images,
                "resources": {
                    "limits": resources_limits,
                    "requests": resources_requests
                },
                "volumeMounts": k8s_volume_mounts,
                "workingDir": working_dir,
                "imagePullPolicy": conf.get('IMAGE_PULL_POLICY','Always')
            },
            "nodeSelector": node_selector,
            "securityContext":{
                "privileged":True if task.job_template.privileged else False
            },
            "affinity": {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "podAffinityTerm": {
                                "labelSelector": {
                                    "matchLabels": {
                                        "pipeline-id": str(pipeline.id)
                                    }
                                },
                                "topologyKey": "kubernetes.io/hostname"
                            },
                            "weight": 80
                        }
                    ]
                }
            },
            "metadata": {
                "labels": pod_label,
                "annotations":pod_annotations
            },
            "retryStrategy": {
                "limit": int(task.retry)
            } if task.retry else None,
            "volumes": k8s_volumes,
            "hostAliases":[
                {
                    "hostnames": [hostname],
                    "ip":hostAliases[hostname]
                } for hostname in hostAliases
            ],
            "timeout":task.timeout if task.timeout else None
        }

        return task_template




    # 配置拉取秘钥
    hubsecret_list = []
    for task_name in all_tasks:
        # 配置拉取秘钥。本来在contain里面，workflow在外面
        task_temp = all_tasks[task_name]
        if task_temp.job_template.images.repository.hubsecret:
            hubsecret = task_temp.job_template.images.repository.hubsecret
            if hubsecret not in hubsecret_list:
                hubsecret_list.append(hubsecret)

    # 设置workflow标签
    if not workflow_label:
        workflow_label={}

    workflow_label['upload-rtx']=g.user.username if g and g.user and g.user.username else pipeline.created_by.username
    workflow_label['run-rtx'] = g.user.username if g and g.user and g.user.username else pipeline.created_by.username
    workflow_label['pipeline-rtx'] = pipeline.created_by.username
    workflow_label['save-time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    workflow_label['pipeline-id'] = str(pipeline.id)
    workflow_label['run-id'] = global_envs.get('KFJ_RUN_ID','')   # 以此来绑定运行时id，不能用kfp的run—id。那个是传到kfp以后才产生的。
    workflow_label['cluster'] = pipeline.project.cluster['NAME']

    containers_template=[]
    for task_name in dag:
        containers_template.append(make_container_template(task_name))

    workflow_json = make_workflow_yaml(pipeline=pipeline,workflow_label=workflow_label,hubsecret_list=hubsecret_list,dag_templates=make_dag_template(),containers_templates=containers_template)

    pipeline_file = template_str(json.dumps(workflow_json,ensure_ascii=False,indent=4))

    return pipeline_file,workflow_label['run-id']

# @pysnooper.snoop(watch_explode=())
def run_pipeline(cluster,workflow_json):
    crd_name = workflow_json.get('metadata',{}).get('name','')
    from myapp.utils.py.py_k8s import K8s
    k8s_client = K8s(cluster.get('KUBECONFIG', ''))
    namespace = workflow_json.get('metadata',{}).get("namespace",conf.get('PIPELINE_NAMESPACE'))
    crd_info = conf.get('CRD_INFO', {}).get('workflow', {})
    try:
        workflow_obj = k8s_client.get_one_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
        if workflow_obj:
            k8s_client.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, name=crd_name)
            time.sleep(1)

        crd = k8s_client.create_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],namespace=namespace, body=workflow_json)
    except Exception as e:
        print(e)

    return crd_name



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
        "pipeline_url":{"type": "ellip2", "width": 400},
        "modified": {"type": "ellip2", "width": 150}
    }
    add_columns = ['project','name','describe']
    edit_columns = ['project', 'name', 'describe', 'schedule_type', 'cron_time', 'depends_on_past', 'max_active_runs','expired_limit', 'parallelism', 'global_env', 'alert_status', 'alert_user', 'parameter','cronjob_start_time']
    show_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','expired_limit','parallelism','global_env','dag_json','pipeline_file','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand','parameter','alert_status','alert_user','cronjob_start_time']
    # show_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    search_columns = ['id', 'created_by', 'name', 'describe', 'schedule_type', 'project']


    base_filters = [["id", Pipeline_Filter, lambda: []]]
    conv = GeneralModelConverter(datamodel)


    add_form_extra_fields = {

        "name": StringField(
            _(datamodel.obj.lab('name')),
            description="英文名(小写字母、数字、- 组成)，最长50个字符",
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
            default='{}',
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
            default='Always',
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
            description="选择通知状态",
            validators=[Length(0,400),]
        ),
        "alert_user": StringField(
            label=_(datamodel.obj.lab('alert_user')),
            widget=BS3TextFieldWidget(),
            description="选择通知用户，每个用户使用逗号分隔"
        )

    }


    edit_form_extra_fields = add_form_extra_fields


    related_views = [Task_ModelView, ]

    def delete_task_run(self,task):
        from myapp.utils.py.py_k8s import K8s
        k8s_client = K8s(task.pipeline.project.cluster.get('KUBECONFIG',''))
        namespace = task.pipeline.namespace
        # 删除运行时容器
        pod_name = "run-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace,pod_name=pod['name'])
            run_id = pod['labels'].get('run-id', '')
            if run_id:
                k8s_client.delete_workflow(all_crd_info = conf.get("CRD_INFO", {}), namespace=namespace,run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id": run_id})
                time.sleep(2)


        # 删除debug容器
        pod_name = "debug-" + task.pipeline.name.replace('_', '-') + "-" + task.name.replace('_', '-')
        pod_name = pod_name.lower()[:60].strip('-')
        pod = k8s_client.get_pods(namespace=namespace, pod_name=pod_name)
        # print(pod)
        if pod:
            pod = pod[0]
        # 有历史，直接删除
        if pod:
            k8s_client.delete_pods(namespace=namespace, pod_name=pod['name'])
            run_id = pod['labels'].get('run-id','')
            if run_id:
                k8s_client.delete_workflow(all_crd_info = conf.get("CRD_INFO", {}), namespace=namespace,run_id=run_id)
                k8s_client.delete_pods(namespace=namespace, labels={"run-id":run_id})
                time.sleep(2)


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


    # 验证args参数,并自动排版dag_json
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
            item.pipeline_file,item.run_id = dag_to_pipeline(item,db.session,workflow_label={"schedule_type":"once"})   # dag_to_pipeline(item,db.session,workflow_label={"schedule_type":"once"})
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

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):
        if not item.project or item.project.type!='org':
            project=db.session.query(Project).filter_by(name='public').filter_by(type='org').first()
            if project:
                item.project = project

        item.name = item.name.replace('_', '-')[0:54].lower().strip('-')
        item.namespace = json.loads(item.project.expand).get('PIPELINE_NAMESPACE', conf.get('PIPELINE_NAMESPACE'))
        # item.alert_status = ','.join(item.alert_status)
        self.pipeline_args_check(item)
        item.create_datetime=datetime.datetime.now()
        item.change_datetime = datetime.datetime.now()
        item.cronjob_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        item.parameter = json.dumps({}, indent=4, ensure_ascii=False)
        # 检测crontab格式
        if item.schedule_type=='crontab':
            if not re.match("^[0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+",item.cron_time):
                item.cron_time=''



    # @pysnooper.snoop()
    def pre_update(self, item):

        if item.expand:
            core.validate_json(item.expand)
            item.expand = json.dumps(json.loads(item.expand),indent=4,ensure_ascii=False)
        else:
            item.expand='{}'
        item.name = item.name.replace('_', '-')[0:54].lower()
        item.namespace = json.loads(item.project.expand).get('PIPELINE_NAMESPACE', conf.get('PIPELINE_NAMESPACE'))
        # item.alert_status = ','.join(item.alert_status)
        self.merge_upstream(item)
        self.pipeline_args_check(item)
        item.change_datetime = datetime.datetime.now()
        if item.parameter:
            item.parameter = json.dumps(json.loads(item.parameter),indent=4,ensure_ascii=False)
        else:
            item.parameter = '{}'

        if (item.schedule_type=='crontab' and self.src_item_json.get("schedule_type")=='once') or (item.cron_time!=self.src_item_json.get("cron_time",'')):
            item.cronjob_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 限制提醒
        if item.schedule_type=='crontab':
            if not re.match("^[0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+ [0-9/*]+",item.cron_time.strip().replace('  ',' ')):
                item.cron_time=''
            if not item.project.node_selector:
                flash('无法保障公共集群的稳定性，定时任务请选择专门的日更集群项目组','warning')
            else:
                org = item.project.node_selector.replace('org=','')
                if not org or org=='public':
                    flash('无法保障公共集群的稳定性，定时任务请选择专门的日更集群项目组','warning')


    def pre_update_web(self,item):
        item.dag_json = item.fix_dag_json()
        item.expand = json.dumps(item.fix_expand(),indent=4,ensure_ascii=False)
        db.session.commit()

    # 删除前先把下面的task删除了，把里面的运行实例也删除了
    # @pysnooper.snoop()
    def pre_delete(self, pipeline):
        db.session.commit()
        if "(废弃)" not in pipeline.describe:
            pipeline.describe+="(废弃)"
        pipeline.schedule_type='once'
        pipeline.expand=""
        pipeline.dag_json="{}"
        db.session.commit()

        back_crds = pipeline.get_workflow()
        self.delete_bind_crd(back_crds)

        tasks = pipeline.get_tasks()
        # 删除task启动的所有实例
        for task in tasks:
            self.delete_task_run(task)

        for task in tasks:
            db.session.delete(task)

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

    # 保存pipeline正在运行的workflow信息
    def save_workflow(self,back_crds):
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
                                        cluster=labels.get("cluster",''),
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


    # @event_logger.log_this
    @expose("/run_pipeline/<pipeline_id>", methods=["GET", "POST"])
    @check_pipeline_perms
    def run_pipeline(self,pipeline_id):
        print(pipeline_id)
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        pipeline.delete_old_task()
        tasks = db.session.query(Task).filter_by(pipeline_id=pipeline_id).all()
        if not tasks:
            flash('no task', 'warning')
            return redirect('/pipeline_modelview/web/%s' % pipeline.id)

        time.sleep(1)

        back_crds = pipeline.get_workflow()
        # self.save_workflow(back_crds)
        # 这里直接删除所有的历史任务流，正在运行的也删除掉
        # not_running_crds = back_crds  # [crd for crd in back_crds if 'running' not in crd['status'].lower()]
        self.delete_bind_crd(back_crds)

        # 删除task启动的所有实例
        for task in tasks:
            self.delete_task_run(task)

        # self.delete_workflow(pipeline)
        pipeline.pipeline_file,pipeline.run_id = dag_to_pipeline(pipeline, db.session,workflow_label={"schedule_type":"once"})  # 合成workflow
        # print('make pipeline file %s' % pipeline.pipeline_file)
        # return
        print('begin upload and run pipeline %s' % pipeline.name)
        pipeline.version_id = ''
        crd_name = run_pipeline(pipeline.project.cluster,json.loads(pipeline.pipeline_file))   # 会根据版本号是否为空决定是否上传
        pipeline.pipeline_argo_id = crd_name
        db.session.commit()  # 更新
        back_crds = pipeline.get_workflow()
        # self.save_workflow(back_crds)

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
        return redirect('/frontend/showOutLink?url=%s'%urllib.parse.quote(url, safe=""))
        # 返回模板
        # return self.render_template('link.html', data=data)


    # # @event_logger.log_this
    @expose("/web/log/<pipeline_id>", methods=["GET"])
    def web_log(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        namespace = pipeline.namespace
        workflow_name = pipeline.pipeline_argo_id
        cluster = pipeline.project.cluster["NAME"]
        url = f'/frontend/commonRelation?backurl=/workflow_modelview/api/web/dag/{cluster}/{namespace}/{workflow_name}'
        return redirect(url)

    # # @event_logger.log_this
    @expose("/web/monitoring/<pipeline_id>", methods=["GET"])
    def web_monitoring(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()
        if pipeline.run_id:
            url = "http://"+pipeline.project.cluster.get('HOST', request.host)+conf.get('GRAFANA_TASK_PATH')+ pipeline.name
            return redirect(url)
        else:
            flash('no running instance','warning')
            return redirect('/pipeline_modelview/web/%s'%pipeline.id)

    # # @event_logger.log_this
    @expose("/web/pod/<pipeline_id>", methods=["GET"])
    def web_pod(self,pipeline_id):
        pipeline = db.session.query(Pipeline).filter_by(id=pipeline_id).first()
        data = {
            "url": "http://" + pipeline.project.cluster.get('HOST', request.host) + conf.get('K8S_DASHBOARD_CLUSTER') + '#/search?namespace=%s&q=%s' % (conf.get('PIPELINE_NAMESPACE'), pipeline.name.replace('_', '-').lower()),
            "target":"div.kd-chrome-container.kd-bg-background",
            "delay":500,
            "loading": True
        }
        # 返回模板
        if pipeline.project.cluster['NAME']==conf.get('ENVIRONMENT'):
            return self.render_template('link.html', data=data)
        else:
            return self.render_template('external_link.html', data=data)

    @expose("/web/runhistory/<pipeline_id>", methods=["GET"])
    def web_runhistory(self,pipeline_id):
        url = conf.get('MODEL_URLS', {}).get('runhistory', '') + '?filter=' + urllib.parse.quote(json.dumps([{"key": "pipeline", "value": int(pipeline_id)}], ensure_ascii=False))
        print(url)
        return redirect(url)


    @expose("/web/workflow/<pipeline_id>", methods=["GET"])
    def web_workflow(self,pipeline_id):
        url = conf.get('MODEL_URLS', {}).get('workflow', '') + '?filter=' + urllib.parse.quote(json.dumps([{"key": "labels", "value": '"pipeline-id": "%s"'%pipeline_id}], ensure_ascii=False))
        print(url)
        return redirect(url)


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


class Pipeline_ModelView(Pipeline_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(Pipeline)
    # base_order = ("changed_on", "desc")
    # order_columns = ['changed_on']

appbuilder.add_view_no_menu(Pipeline_ModelView)


# 添加api
class Pipeline_ModelView_Api(Pipeline_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(Pipeline)
    route_base = '/pipeline_modelview/api'
    # show_columns = ['project','name','describe','namespace','schedule_type','cron_time','node_selector','depends_on_past','max_active_runs','parallelism','global_env','dag_json','pipeline_file_html','pipeline_argo_id','version_id','run_id','created_by','changed_by','created_on','changed_on','expand']
    list_columns = ['id','project','pipeline_url','creator','modified']
    add_columns = ['project','name','describe']
    edit_columns = ['project','name','describe','schedule_type','cron_time','depends_on_past','max_active_runs','parallelism','dag_json','global_env','alert_status','alert_user','expand','parameter','cronjob_start_time']

    related_views = [Task_ModelView_Api,]

    def pre_add_web(self):
        self.default_filter = {
            "created_by": g.user.id
        }

    add_form_query_rel_fields = {
        "project": [["name", Project_Join_Filter, 'org']]
    }
    edit_form_query_rel_fields=add_form_query_rel_fields


appbuilder.add_api(Pipeline_ModelView_Api)




