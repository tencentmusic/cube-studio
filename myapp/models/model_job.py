from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey,Float
from sqlalchemy.orm import relationship
import datetime,time,json
from sqlalchemy import (
    Boolean,
    Column,
    create_engine,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Enum,
)
import numpy
import random
import copy
import logging
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from .model_team import Project
from myapp import app,db
from myapp.models.helpers import ImportMixin
# from myapp.models.base import MyappModel
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config
from myapp.utils import core
import re
from myapp.utils.py import py_k8s
import pysnooper

# 定义model
class Repository(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'repository'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique = True, nullable=False)
    server = Column(String(100), nullable=False)
    user = Column(String(100), nullable=False)
    password = Column(String(100), nullable=False)
    hubsecret = Column(String(100))

    def __repr__(self):
        return self.name

    label_columns_spec={
        "server":'域名',
        "user":"用户名",
        "hubsecret": 'k8s hubsecret',
    }
    label_columns=MyappModelBase.label_columns.copy()
    label_columns.update(label_columns_spec)

# 定义model
class Images(Model,AuditMixinNullable,MyappModelBase):
    __tablename__='images'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )

    name = Column(String(200), nullable=False)
    describe = Column(Text)
    repository_id = Column(Integer, ForeignKey('repository.id'))    # 定义外键
    repository = relationship(
        "Repository", foreign_keys=[repository_id]
    )
    entrypoint=Column(String(200))
    dockerfile=Column(Text)
    gitpath=Column(String(200))

    label_columns_spec={
        "project":'功能分类',
    }
    label_columns = MyappModelBase.label_columns.copy()
    label_columns.update(label_columns_spec)

    @property
    def images_url(self):
        if self.gitpath:
            return Markup(f'<a href="{self.gitpath}">{self.name}</a>')
        return self.name

    def __repr__(self):
        return self.name


# 定义model
class Job_Template(Model,AuditMixinNullable,MyappModelBase):
    __tablename__='job_template'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(100), nullable=False,unique=True)
    version = Column(Enum('Release','Alpha'),nullable=False,default='Release')
    images_id = Column(Integer, ForeignKey('images.id'))  # 定义外键
    images = relationship(
        Images, foreign_keys=[images_id]
    )
    hostAliases = Column(Text)   # host文件
    describe = Column(Text)
    workdir=Column(String(400))
    entrypoint=Column(String(200))
    args=Column(Text)
    env = Column(Text)   # 默认自带的环境变量
    volume_mount = Column(String(400),default='')  # 强制必须挂载
    privileged = Column(Boolean, default=False)   # 是否启用特权模式
    accounts = Column(String(100))   # 使用账户
    demo=Column(Text)
    expand = Column(Text(65536), default='{}')

    label_columns_spec={
        "project": "功能分类",
    }
    label_columns = MyappModelBase.label_columns.copy()
    label_columns.update(label_columns_spec)

    def __repr__(self):
        return self.name   # +"(%s)"%self.version

    @renders('args')
    def args_html(self):
        return Markup('<pre><code>' + self.args + '</code></pre>')

    @renders('demo')
    def demo_html(self):
        return Markup('<pre><code>' + self.demo + '</code></pre>')

    @renders('expand')
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')


    @renders('name')
    def name_title(self):
        return Markup(f'<a data-toggle="tooltip" rel="tooltip" title data-original-title="{self.describe}">{self.name}</a>')


    @property
    def images_url(self):
        return Markup(f'<a target=_blank href="/images_modelview/show/{self.images.id}">{self.images.name}</a>')

    # import pysnooper
    # @pysnooper.snoop()
    def get_env(self,name):
        if self.env and name in self.env:
            envs = self.env.split('\n')
            for env in envs:
                if name in env:
                    return env[env.index('=')+1:].strip()
        else:
            return None


    def clone(self):
        return Job_Template(
            name=self.name,
            version=self.version,
            project_id=self.project_id,
            images_id=self.images_id,
            describe=self.describe,
            args=self.args,
            demo=self.demo,
            expand=self.expand
        )


# 定义model
class Pipeline(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'pipeline'
    id = Column(Integer, primary_key=True)
    name = Column(String(100),nullable=False,unique=True)
    describe = Column(String(200),nullable=False)
    project_id = Column(Integer, ForeignKey('project.id'),nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    dag_json = Column(Text,nullable=False,default='{}')
    namespace=Column(String(100),default='pipeline')
    global_env = Column(String(500),default='')
    schedule_type = Column(Enum('once', 'crontab'),nullable=False,default='once')
    cron_time = Column(String(100))        # 调度周期
    pipeline_file=Column(Text(65536),default='')
    pipeline_argo_id = Column(String(100))
    version_id = Column(String(100))
    run_id = Column(String(100))
    node_selector = Column(String(100), default='cpu=true,train=true')  # 挂载
    image_pull_policy = Column(Enum('Always','IfNotPresent'),nullable=False,default='Always')
    parallelism = Column(Integer, nullable=False,default=1)  # 同一个pipeline，最大并行的task数目
    alert_status = Column(String(100), default='Pending,Running,Succeeded,Failed,Terminated')   # 哪些状态会报警Pending,Running,Succeeded,Failed,Unknown,Waiting,Terminated
    alert_user = Column(String(300), default='')
    expand = Column(Text(65536),default='[]')
    depends_on_past = Column(Boolean, default=False)
    max_active_runs = Column(Integer, nullable=False,default=3)   # 最大同时运行的pipeline实例
    expired_limit = Column(Integer, nullable=False, default=1)  # 过期保留个数，此数值有效时，会优先使用，覆盖max_active_runs的功能
    parameter = Column(Text(65536), default='{}')

    def __repr__(self):
        return self.name

    @property
    def pipeline_url(self):
        pipeline_url="/pipeline_modelview/web/" +str(self.id)
        return Markup(f'<a href="{pipeline_url}">{self.describe}</a>')

    @property
    def run_pipeline(self):
        pipeline_run_url = "/pipeline_modelview/run_pipeline/" +str(self.id)
        return Markup(f'<a target=_blank href="{pipeline_run_url}">run</a>')


    @property
    def cronjob_start_time(self):
        cronjob_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.parameter:
            return json.loads(self.parameter).get('cronjob_start_time',cronjob_start_time)
        return cronjob_start_time


    @property
    def log(self):
        if self.run_id:
            pipeline_url = "/pipeline_modelview/web/log/%s"%self.id
            return Markup(f'<a target=_blank href="{pipeline_url}">日志</a>')
        else:
            return Markup(f'日志')


    @property
    def pod(self):
        url = "/pipeline_modelview/web/pod/%s" % self.id
        return Markup(f'<a target=_blank href="{url}">pod</a>')



    @renders('dag_json')
    def dag_json_html(self):
        dag_json = self.dag_json or '{}'
        return Markup('<pre><code>' + dag_json + '</code></pre>')


    @renders('expand')
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    @renders('parameter')
    def parameter_html(self):
        return Markup('<pre><code>' + self.parameter + '</code></pre>')


    @renders('pipeline_file')
    def pipeline_file_html(self):
        pipeline_file = self.pipeline_file or ''
        return Markup('<pre><code>' + pipeline_file + '</code></pre>')

    # @renders('describe')
    # def describe_html(self):
    #     return Markup('<pre><code>' + self.pipeline_file + '</code></pre>')

    # 获取pipeline中的所有task
    def get_tasks(self,dbsession=db.session):
        return dbsession.query(Task).filter_by(pipeline_id=self.id).all()

    # @pysnooper.snoop()
    def delete_old_task(self, dbsession=db.session):
        try:
            expand_tasks = json.loads(self.expand) if self.expand else []
            tasks = dbsession.query(Task).filter_by(pipeline_id=self.id).all()
            tasks_id = [int(expand_task['id']) for expand_task in expand_tasks if expand_task.get('id', '').isdecimal()]
            for task in tasks:
                if task.id not in tasks_id:
                    db.session.delete(task)
                    db.session.commit()
        except Exception as e:
            print(e)

    # 获取当期运行时workflow的数量
    def get_workflow(self):

        back_crds = []
        try:
            k8s_client = py_k8s.K8s(self.project.cluster.get('KUBECONFIG',''))
            crd_info = conf.get("CRD_INFO", {}).get('workflow', {})
            if crd_info:
                crds = k8s_client.get_crd(group=crd_info['group'], version=crd_info['version'],
                                          plural=crd_info['plural'], namespace=self.namespace,
                                          label_selector="pipeline-id=%s"%str(self.id))
                for crd in crds:
                    if crd.get('labels', '{}'):
                        labels = json.loads(crd['labels'])
                        if labels.get('pipeline-id', '') == str(self.id):
                            back_crds.append(crd)
            return back_crds
        except Exception as e:
            print(e)
        return back_crds


    @property
    def run_instance(self):
        # workflow = db.session.query(Workflow).filter_by(foreign_key= str(self.id)).filter_by(status= 'Running').filter_by(create_time > datetime.datetime.now().strftime("%Y-%m-%d")).all()
        # workflow_num = len(workflow) if workflow else 0
        # url = '/workflow_modelview/list/?_flt_2_name=%s'%self.name.replace("_","-")[:54]
        url = r'/workflow_modelview/list/?_flt_2_labels="pipeline-id"%3A+"'+'%s"' % self.id
        # print(url)
        return Markup(f"<a href='{url}'>{self.schedule_type}</a>")  # k8s有长度限制

    # 这个dag可能不对，所以要根据真实task纠正一下
    def fix_dag_json(self,dbsession=db.session):
        if not self.dag_json:
            return "{}"
        dag = json.loads(self.dag_json)
        # 如果添加了task，但是没有保存pipeline，就自动创建dag
        if not dag:
            tasks = self.get_tasks(dbsession)
            if tasks:
                dag = {}
                for task in tasks:
                    dag[task.name] = {}
                dag_json = json.dumps(dag, indent=4, ensure_ascii=False)
                return dag_json
            else:
                return "{}"

        # 清理dag中不存在的task
        if dag:
            tasks = self.get_tasks(dbsession)
            all_task_names = [task.name for task in tasks]
            # 先把没有加入的task加入到dag
            for task in tasks:
                if task.name not in dag:
                    dag[task.name] = {}

            # 把已经删除了的task移除dag
            dag_back = copy.deepcopy(dag)
            for dag_task_name in dag_back:
                if dag_task_name not in all_task_names:
                    del dag[dag_task_name]

            # 将已经删除的task从其他task的上游依赖中删除
            for dag_task_name in dag:
                upstream_tasks = dag[dag_task_name]['upstream'] if 'upstream' in dag[dag_task_name] else []
                new_upstream_tasks = []
                for upstream_task in upstream_tasks:
                    if upstream_task in all_task_names:
                        new_upstream_tasks.append(upstream_task)

                dag[dag_task_name]['upstream'] = new_upstream_tasks



            # def get_downstream(dag):
            #     # 生成下行链路图
            #     for task_name in dag:
            #         dag[task_name]['downstream'] = []
            #         for task_name1 in dag:
            #             if task_name in dag[task_name1].get("upstream", []):
            #                 dag[task_name]['downstream'].append(task_name1)
            #     return dag
            #
            # dag = get_downstream(dag)
            dag_json = json.dumps(dag, indent=4, ensure_ascii=False)

            return dag_json

    # 自动聚焦到视图中央
    # @pysnooper.snoop()
    def fix_position(self):
        expand_tasks = json.loads(self.expand) if self.expand else []
        if not expand_tasks:
            expand_tasks = []
        x=[]
        y=[]
        for item in expand_tasks:
            if "position" in item:
                if item['position'].get('x',0):
                    x.append(int(item['position'].get('x',0)))
                    y.append(int(item['position'].get('y', 0)))
        x_dist=400- numpy.mean(x) if x else 0
        y_dist = 300 -numpy.mean(y) if y else 0
        for item in expand_tasks:
            if "position" in item:
                if item['position'].get('x', 0):
                    item['position']['x'] = int(item['position']['x'])+x_dist
                    item['position']['y'] = int(item['position']['y']) + y_dist

        return expand_tasks




    # 生成前端锁需要的扩展字段
    def fix_expand(self,dbsession=db.session):
        tasks_src = self.get_tasks(dbsession)
        tasks = {}
        for task in tasks_src:
            tasks[str(task.id)] = task

        expand_tasks = json.loads(self.expand) if self.expand else []
        if not expand_tasks:
            expand_tasks=[]
        expand_copy = copy.deepcopy(expand_tasks)

        # 已经不存在的task要删掉
        for item in expand_copy:
            if "data" in item:
                if item['id'] not in tasks:
                    expand_tasks.remove(item)
            else:
                # if item['source'] not in tasks or item['target'] not in tasks:
                expand_tasks.remove(item)   # 删除所有的上下游关系，后面全部重新

        # 增加新的task的位置
        for task_id in tasks:
            exist=False
            for item in expand_tasks:
                if "data" in item and item['id']==str(task_id):
                    exist=True
                    break
            if not exist:
                # if task_id not in expand_tasks:
                expand_tasks.append({
                    "id": str(task_id),
                    "type": "dataSet",
                    "position": {
                        "x": random.randint(100,1000),
                        "y": random.randint(100,1000)
                    },
                    "data": {
                        "taskId": task_id,
                        "taskName": tasks[task_id].name,
                        "name": tasks[task_id].name,
                        "describe": tasks[task_id].label
                    }
                })

        # 重写所有task的上下游关系
        dag_json = json.loads(self.dag_json)
        for task_name in dag_json:
            upstreams = dag_json[task_name].get("upstream", [])
            if upstreams:
                for upstream_name in upstreams:
                    upstream_task_id = [task_id for task_id in tasks if tasks[task_id].name==upstream_name][0]
                    task_id = [task_id for task_id in tasks if tasks[task_id].name==task_name][0]
                    if upstream_task_id and task_id:
                        expand_tasks.append(
                            {
                                "source": str(upstream_task_id),
                                # "sourceHandle": None,
                                "target": str(task_id),
                                # "targetHandle": None,
                                "id": self.name + "__edge-%snull-%snull" % (upstream_task_id, task_id)
                            }
                        )
        return expand_tasks


    # @pysnooper.snoop()
    def clone(self):
        return Pipeline(
            name=self.name.replace('_','-'),
            project_id=self.project_id,
            dag_json=self.dag_json,
            describe=self.describe,
            namespace=self.namespace,
            global_env=self.global_env,
            schedule_type='once',
            cron_time=self.cron_time,
            pipeline_file='',
            pipeline_argo_id=self.pipeline_argo_id,
            node_selector=self.node_selector,
            image_pull_policy=self.image_pull_policy,
            parallelism=self.parallelism,
            alert_status='',
            expand=self.expand,
            parameter=self.parameter
        )


# 定义model
class Task(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    label = Column(String(100), nullable=False)   # 别名
    job_template_id = Column(Integer, ForeignKey('job_template.id'))  # 定义外键
    job_template = relationship(
        "Job_Template", foreign_keys=[job_template_id]
    )
    pipeline_id = Column(Integer, ForeignKey('pipeline.id'))  # 定义外键
    pipeline = relationship(
        "Pipeline", foreign_keys=[pipeline_id]
    )
    working_dir = Column(String(1000),default='')
    command = Column(String(1000),default='')
    overwrite_entrypoint = Column(Boolean,default=False)   # 是否覆盖入口
    args = Column(Text)
    volume_mount = Column(String(200),default='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives')   # 挂载
    node_selector = Column(String(100),default='cpu=true,train=true')   # 挂载
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')
    timeout = Column(Integer, nullable=False,default=0)
    retry = Column(Integer, nullable=False,default=0)
    outputs = Column(Text,default='{}')   # task的输出，会将输出复制到minio上   {'prediction': '/output.txt'}
    monitoring = Column(Text,default='{}')  # 该任务的监控信息
    expand = Column(Text(65536), default='')
    # active = Column(Boolean,default=True)  # 是否激活，可以先配置再运行跑
    export_parent = "pipeline"


    def __repr__(self):
        return self.name

    @property
    def debug(self):
        return Markup(f'<a target=_blank href="/task_modelview/debug/{self.id}">debug</a>')

    @property
    def run(self):
        return Markup(f'<a target=_blank href="/task_modelview/run/{self.id}">run</a>')

    @property
    def clear(self):
        return Markup(f'<a href="/task_modelview/clear/{self.id}">clear</a>')

    @property
    def log(self):
        return Markup(f'<a target=_blank href="/task_modelview/log/{self.id}">log</a>')

    def get_node_selector(self):
        project_node_selector = self.get_default_node_selector(self.pipeline.project.node_selector,self.resource_gpu,'train')
        gpu_type = core.get_gpu(self.resource_gpu)[1]
        if gpu_type:
            project_node_selector+=',gpu-type='+gpu_type
        return project_node_selector


    @renders('args')
    def args_html(self):
        return Markup('<pre><code>' + self.args + '</code></pre>')

    @renders('expand')
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    @renders('monitoring')
    def monitoring_html(self):
        try:
            monitoring = json.loads(self.monitoring)
            monitoring['link']=self.pipeline.project.cluster.get('GRAFANA_HOST','').strip('/')+conf.get('GRAFANA_TASK_PATH')+monitoring.get('pod_name','')
            return Markup('<pre><code>' + json.dumps(monitoring,ensure_ascii=False,indent=4) + '</code></pre>')
        except Exception as e:
            return Markup('<pre><code> 暂无 </code></pre>')

    @property
    def job_args_demo(self):
        return Markup('<pre><code>' + self.job_template.demo + '</code></pre>')

    @property
    def job_template_url(self):
        return Markup(f'<a target=_blank href="/job_template_modelview/show/{self.job_template.id}">{self.job_template.name}</a>')



    def clone(self):
        return Task(
            name=self.name.replace('_','-'),
            label=self.label,
            job_template_id=self.job_template_id,
            pipeline_id=self.pipeline_id,
            working_dir=self.working_dir,
            command=self.command,
            args=self.args,
            volume_mount=self.volume_mount,
            node_selector=self.node_selector,
            resource_memory=self.resource_memory,
            resource_cpu=self.resource_cpu,
            timeout=self.timeout,
            retry=self.retry,
            expand=self.expand
        )


# 每次上传运行
class RunHistory(Model,MyappModelBase):
    __tablename__ = "run"
    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey('pipeline.id'))  # 定义外键
    pipeline = relationship(
        "Pipeline", foreign_keys=[pipeline_id]
    )
    pipeline_file = Column(Text(65536), default='')
    pipeline_argo_id = Column(String(100))   # 上传的pipeline id
    version_id = Column(String(100))        # 上传的版本号
    experiment_id = Column(String(100))
    run_id = Column(String(100))
    message = Column(Text, default='')
    created_on = Column(DateTime, default=datetime.datetime.now, nullable=False)
    execution_date=Column(String(100), nullable=False)
    status = Column(String(100),default='comed')   # commed表示已经到了该调度的时间，created表示已经发起了调度。注意操作前校验去重


    @property
    def status_url(self):
        if self.status=='comed':
            return self.status
        return Markup(f'<a target=_blank href="/workflow_modelview/list/?_flt_2_labels={self.run_id}">{self.status}</a>')

    @property
    def creator(self):
        return self.pipeline.creator

    @property
    def pipeline_url(self):
        return Markup(f'<a target=_blank href="/pipeline_modelview/web/{self.pipeline.id}">{self.pipeline.describe}</a>')


    @property
    def history(self):
        url = r'/workflow_modelview/list/?_flt_2_labels="pipeline-id"%3A+"' + '%s"' % self.pipeline_id
        return Markup(f"<a href='{url}'>运行记录</a>")

    @property
    def log(self):
        if self.run_id:
            pipeline_url = self.pipeline.project.cluster.get('PIPELINE_URL')+ "runs/details/" +str(self.run_id)
            return Markup(f'<a target=_blank href="{pipeline_url}">日志</a>')
        else:
            return Markup(f'日志')

import sqlalchemy as sa
class Crd:
    # __tablename__ = "crd"
    id = Column(Integer, primary_key=True)
    name = Column(String(100),default='')
    namespace = Column(String(100), default='')
    create_time=Column(String(100), default='')
    change_time = Column(String(100), default='')

    status = Column(String(100), default='')
    annotations = Column(Text, default='')
    labels = Column(Text, default='')
    spec = Column(Text(65536), default='')
    status_more = Column(Text(65536), default='')
    username = Column(String(100), default='')
    info_json = Column(Text, default='{}')
    add_row_time = Column(DateTime, default=datetime.datetime.now)
    # delete = Column(Boolean,default=False)
    foreign_key = Column(String(100), default='')

    @renders('annotations')
    def annotations_html(self):
        return Markup('<pre><code>' + self.annotations + '</code></pre>')

    @renders('labels')
    def labels_html(self):
        return Markup('<pre><code>' + self.labels + '</code></pre>')

    @property
    def final_status(self):
        status='未知'
        try:
            if self.status_more:
                status = json.loads(self.status_more).get('phase','未知')
        except Exception as e:
            print(e)
        return status

    @renders('spec')
    def spec_html(self):
        return Markup('<pre><code>' + self.spec + '</code></pre>')

    @renders('status_more')
    def status_more_html(self):
        return Markup('<pre><code>' + self.status_more + '</code></pre>')

    @renders('info_json')
    def info_json_html(self):
        return Markup('<pre><code>' + self.info_json + '</code></pre>')

    @renders('namespace')
    def namespace_url(self):
        # user_roles = [role.name.lower() for role in list(g.user.roles)]
        # if "admin" in user_roles:
        url = conf.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (self.namespace, self.name.replace('_', '-'))
        # else:
        #     url = conf.get('K8S_DASHBOARD_PIPELINE','')+'#/search?namespace=%s&q=%s'%(self.namespace,self.name.replace('_','-'))
        return Markup(f'<a target=_blank href="{url}">{self.namespace}</a>')

    @property
    def stop(self):
        return Markup(f'<a href="../stop/{self.id}">停止</a>')

# 定义model
class Workflow(Model,Crd,MyappModelBase):
    __tablename__ = 'workflow'

    @renders('namespace')
    def namespace_url(self):
        if self.pipeline:
            url = conf.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (self.namespace, self.pipeline.name.replace('_', '-'))
            return Markup(f'<a target=_blank href="{url}">{self.namespace}</a>')
        else:
            url = conf.get('K8S_DASHBOARD_CLUSTER', '') + '#/search?namespace=%s&q=%s' % (self.namespace, self.name.replace('_', '-'))
            return Markup(f'<a target=_blank href="{url}">{self.namespace}</a>')

    @property
    def run_history(self):
        label = json.loads(self.labels) if self.labels else {}
        runid = label.get('run-id','')
        if runid:
            return db.session.query(RunHistory).filter(RunHistory.pipeline_file.contains(runid)).first()
            # return db.session.query(RunHistory).filter_by(run_id=runid).first()
        else:
            return None

    @property
    def schedule_type(self):
        run_history = self.run_history
        if run_history:
            return 'crontab'
        else:
            return 'once'


    @property
    def execution_date(self):
        run_history = self.run_history
        if run_history:
            return run_history.execution_date
        else:
            return 'once'


    @property
    def task_status(self):
        status_mode = json.loads(self.status_more)
        task_status={}
        nodes=status_mode.get('nodes',{})
        tasks = self.pipeline.get_tasks()
        for pod_name in nodes:
            pod = nodes[pod_name]
            if pod['type']=='Pod':
                if pod['phase']=='Succeeded':     # 那些重试和失败的都忽略掉
                    templateName=pod['templateName']
                    for task in tasks:
                        if task.name==templateName:
                            finish_time = datetime.datetime.strptime(pod['finishedAt'], '%Y-%m-%d %H:%M:%S')
                            start_time = datetime.datetime.strptime(pod['startedAt'], '%Y-%m-%d %H:%M:%S')
                            elapsed = (finish_time - start_time).days * 24 + (finish_time - start_time).seconds / 60 / 60
                            task_status[task.label]= str(round(elapsed,2))+"h"

        message=""
        for key in task_status:
            message += key+": "+task_status[key]+"\n"
        return Markup('<pre><code>' + message + '</code></pre>')



    @property
    def elapsed_time(self):
        status_mode = json.loads(self.status_more)
        finish_time=status_mode.get('finishedAt',self.change_time)
        if not finish_time: finish_time=self.change_time
        start_time = status_mode.get('startedAt', '')
        try:
            if finish_time and start_time:
                if 'T' in finish_time:
                    finish_time = datetime.datetime.strptime(finish_time,'%Y-%m-%dT%H:%M:%S')
                else:
                    finish_time = datetime.datetime.strptime(finish_time, '%Y-%m-%d %H:%M:%S')
                if 'T' in start_time:
                    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
                else:
                    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                elapsed = (finish_time-start_time).days*24+(finish_time-start_time).seconds/60/60
                return str(round(elapsed,2))+"h"
        except Exception as e:
            print(e)
        return '未知'


    @property
    def pipeline_url(self):
        if self.labels:
            try:
                labels = json.loads(self.labels)
                pipeline_id = labels.get("pipeline-id",'')
                if pipeline_id:
                    pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()
                    if pipeline:
                        # return Markup(f'{pipeline.describe}')
                        return Markup(f'<a href="/pipeline_modelview/web/{pipeline.id}">{pipeline.describe}</a>')

                pipeline_name = self.name[:-6]
                pipeline = db.session.query(Pipeline).filter_by(name=pipeline_name).first()
                if pipeline:
                    return Markup(f'{pipeline.describe}')

            except Exception as e:
                print(e)
        return Markup(f'未知')

    @property
    def pipeline(self):
        if self.labels:
            try:
                labels = json.loads(self.labels)
                pipeline_id = labels.get("pipeline-id",'')
                if pipeline_id:
                    pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()
                    if pipeline:
                        return pipeline

                # pipeline_name = self.name[:-6]
                # pipeline = db.session.query(Pipeline).filter_by(name=pipeline_name).first()
                # return pipeline

            except Exception as e:
                print(e)
        return None


    @property
    def project(self):
        pipeline = self.pipeline
        if pipeline:
            return pipeline.project.name
        else:
            return "未知"

    @property
    def log(self):
        if self.labels:
            try:
                labels = json.loads(self.labels)
                run_id = labels.get("pipeline/runid",'')
                if run_id:
                    pipeline_url = conf.get('PIPELINE_URL')+ "runs/details/" +str(run_id)
                    return Markup(f'<a target=_blank href="{pipeline_url}">日志</a>')
            except Exception as e:
                print(e)

        return Markup(f'日志')


# 定义model
class Tfjob(Model,Crd,MyappModelBase):
    __tablename__ = 'tfjob'

    @property
    def pipeline(self):
        if self.labels:
            try:
                labels = json.loads(self.labels)
                pipeline_id = labels.get("pipeline-id",'')
                if pipeline_id:
                    pipeline = db.session.query(Pipeline).filter_by(id=int(pipeline_id)).first()
                    return Markup(f'<a href="/pipeline_modelview/list/?_flt_2_name={pipeline.name}">{pipeline.describe}</a>')
            except Exception as e:
                print(e)
        return Markup(f'未知')

    @property
    def run_instance(self):
        if self.labels:
            try:
                labels = json.loads(self.labels)
                run_id = labels.get("run-id",'')
                if run_id:
                    return Markup(f'<a href="/workflow_modelview/list/?_flt_2_labels={run_id}">运行实例</a>')
            except Exception as e:
                print(e)
        return Markup(f'未知')


# 定义model
class Xgbjob(Model,Crd,MyappModelBase):
    __tablename__ = 'xgbjob'


# 定义model
class Pytorchjob(Model,Crd,MyappModelBase):
    __tablename__ = 'pytorchjob'