from flask_appbuilder import Model
from sqlalchemy.orm import relationship
import json
from sqlalchemy import (
    Text,
    Enum,
)
import numpy
import random
import copy
from myapp.models.helpers import AuditMixinNullable

from myapp import app,db
from myapp.models.helpers import ImportMixin

from sqlalchemy import Column, Integer, String, ForeignKey
from flask_appbuilder.models.decorators import renders
from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config



class Service_Pipeline(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'service_pipeline'
    id = Column(Integer, primary_key=True)
    name = Column(String(100),nullable=False,unique=True)
    describe = Column(String(200),nullable=False)
    project_id = Column(Integer, ForeignKey('project.id'),nullable=False)
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    dag_json = Column(Text,nullable=False,default='{}')
    namespace=Column(String(100),default='service')
    env = Column(String(500),default='')
    run_id = Column(String(100))
    node_selector = Column(String(100), default='cpu=true,train=true')
    images = Column(String(200), nullable=False)
    working_dir = Column(String(100),default='')
    command = Column(String(1000),default='')
    volume_mount = Column(String(200),default='')
    image_pull_policy = Column(Enum('Always','IfNotPresent'),nullable=False,default='Always')
    replicas = Column(Integer, default=1)  
    resource_memory = Column(String(100),default='2G')
    resource_cpu = Column(String(100), default='2')
    resource_gpu= Column(String(100), default='0')

    parallelism = Column(Integer, nullable=False,default=1)  # 同一个service_pipeline，最大并行的task数目
    alert_status = Column(String(100), default='Pending,Running,Succeeded,Failed,Terminated')   # 哪些状态会报警Pending,Running,Succeeded,Failed,Unknown,Waiting,Terminated
    alert_user = Column(String(300), default='')
    expand = Column(Text(65536),default='[]')
    parameter = Column(Text(65536), default='{}')

    def __repr__(self):
        return self.name

    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'service')


    @property
    def service_pipeline_url(self):
        service_pipeline_url="/service_pipeline_modelview/web/" +str(self.id)
        return Markup(f'<a href="{service_pipeline_url}">{self.describe}</a>')

    @property
    def run(self):
        service_pipeline_run_url = "/service_pipeline_modelview/run_service_pipeline/" +str(self.id)
        return Markup(f'<a target=_blank href="{service_pipeline_run_url}">运行</a>')

    @property
    def log(self):
        if self.run_id:
            service_pipeline_url = "/service_pipeline_modelview/web/log/%s"%self.id
            return Markup(f'<a target=_blank href="{service_pipeline_url}">日志</a>')
        else:
            return Markup('日志')


    @property
    def pod(self):
        url = "/service_pipeline_modelview/web/pod/%s" % self.id
        return Markup(f'<a target=_blank href="{url}">pod</a>')


    @property
    def operate_html(self):
        dom=f'''
        <a target=_blank href="/service_pipeline_modelview/run_service_pipeline/{self.id}">部署</a> | 
        <a target=_blank href="/service_pipeline_modelview/web/pod/{self.id}">pod</a> | 
        <a target=_blank href="/service_pipeline_modelview/web/log/{self.id}">日志</a> |
        <a target=_blank href="/service_pipeline_modelview/web/monitoring/{self.id}">监控</a> |
        <a href="/service_pipeline_modelview/clear/{self.id}">清理</a>
        '''
        return Markup(dom)


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


    def get_root_node_name(self):
        roots = []
        dag_json = json.loads(self.dag_json)
        for node_name in dag_json:
            if dag_json[node_name].get('template-group', '') == 'endpoint':
                roots.append(node_name)
        return roots


    # 产生每个节点的下游节点
    def make_downstream(self,dag_json):
        for node_name in dag_json:
            if 'downstream' not in dag_json[node_name]:
                dag_json[node_name]['downstream'] = []
            node = dag_json[node_name]
            upstream_nodes = node.get('upstream', [])
            for upstream_node in upstream_nodes:
                if upstream_node in dag_json:
                    dag_json[upstream_node]['downstream'] = list(
                        set(dag_json[upstream_node]['downstream'] + [node_name])) if 'downstream' in dag_json[
                        upstream_node] else [node_name]

            dag_json[node_name]['trace'] = {}
        return dag_json

    # 产生每个节点的上游节点
    def make_upstream(self,dag_json):
        for node_name in dag_json:
            if 'upstream' not in dag_json[node_name]:
                dag_json[node_name]['upstream'] = []

            node = dag_json[node_name]
            down_nodes = node.get('downstream', [])
            for down_node in down_nodes:
                if down_node in dag_json:
                    dag_json[down_node]['upstream'] = list(
                        set(dag_json[down_node]['upstream'] + [node_name])) if 'upstream' in dag_json[down_node] else [
                        node_name]

            dag_json[down_node]['trace'] = {}

        return dag_json


    # 这个dag可能不对，所以要根据真实task纠正一下
    def fix_dag_json(self,dbsession=db.session):
        if not self.dag_json:
            return "{}"
        dag = json.loads(self.dag_json)
        # 如果添加了task，但是没有保存service_pipeline，就自动创建dag
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

