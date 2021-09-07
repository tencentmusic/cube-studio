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
from myapp.utils import core
import re
from myapp.models.base import MyappModelBase
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from myapp import app,db
from myapp.models.helpers import ImportMixin
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config


# 定义model
class Hyperparameter_Tuning(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'hp'
    id = Column(Integer, primary_key=True)
    job_type = Column(Enum('Job','TFJob','XGBJob','PyTorchJob'),nullable=False,default='Job')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(200), unique = True, nullable=False)
    namespace = Column(String(200), nullable=False,default='katib')
    describe = Column(Text)
    parallel_trial_count = Column(Integer,default=3)
    max_trial_count = Column(Integer,default=12)
    max_failed_trial_count = Column(Integer,default=3)
    objective_type = Column(Enum('maximize','minimize'),nullable=False,default='maximize')
    objective_goal = Column(Float, nullable=False,default=0.99)
    objective_metric_name = Column(String(200), nullable=False,default='Validation-accuracy')
    objective_additional_metric_names = Column(String(200),default='')   # 逗号分隔
    algorithm_name = Column(Enum('grid','random','hyperband','bayesianoptimization'),nullable=False,default='random')
    algorithm_setting = Column(Text,default='')  # 搜索算法的配置
    parameters=Column(Text,default='{}')  # 搜索超参的配置
    job_json = Column(Text, default='{}')  # 根据不同算法和参数写入的task模板
    trial_spec=Column(Text,default='')    # 根据不同算法和参数写入的task模板
    working_dir = Column(String(200), default='')  # 挂载
    volume_mount = Column(String(100), default='')  # 挂载
    node_selector = Column(String(100), default='cpu=true,train=true')  # 挂载
    image_pull_policy = Column(Enum('Always', 'IfNotPresent'), nullable=False, default='Always')
    resource_memory = Column(String(100), default='1G')
    resource_cpu = Column(String(100), default='1')

    experiment=Column(Text,default='')  # 构建出来的实验体
    alert_status = Column(String(100), default='')   # 哪些状态会报警Pending,Running,Succeeded,Failed,Unknown,Waiting,Terminated


    def __repr__(self):
        return self.name

    @renders('parameters')
    def parameters_html(self):
        return Markup('<pre><code>' + self.parameters + '</code></pre>')


# '''
# "\"单反斜杠  %5C
# "|"      %7C
# 回车  %0D%0A
# 空格  %20
# 双引号 %22
# "&" %26
# '''
    @property
    def name_url(self):
        return Markup(f'<a target=_blank href="/experiments_modelview/list/?_flt_2_labels=%22{self.name}%22">{self.name}</a>')

    @property
    def describe_url(self):
        return Markup(f'<a target=_blank href="/experiments_modelview/list/?_flt_2_labels=%22{self.name}%22">{self.describe}</a>')

    @property
    def run_url(self):
        return Markup(f'<a href="/hyperparameter_tuning_modelview/create_experiment/{self.id}">run</a>')



    @renders('trial_spec')
    def trial_spec_html(self):
        return Markup('<pre><code>' + self.trial_spec + '</code></pre>')


    @renders('experiment')
    def experiment_html(self):
        return Markup('<pre><code>' + self.experiment + '</code></pre>')

    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'train')


    def clone(self):
        return Hyperparameter_Tuning(
            name=self.name.replace('_','-'),
            job_type = self.job_type,
            describe=self.describe,
            namespace=self.namespace,
            project_id=self.project_id,
            parallel_trial_count=self.parallel_trial_count,
            max_trial_count=self.max_trial_count,
            max_failed_trial_count=self.max_failed_trial_count,
            objective_type=self.objective_type,
            objective_goal=self.objective_goal,
            objective_metric_name=self.objective_metric_name,
            objective_additional_metric_names=self.objective_additional_metric_names,
            algorithm_name=self.algorithm_name,
            algorithm_setting=self.algorithm_setting,
            parameters=self.parameters,
            job_json = self.job_json,
            trial_spec=self.trial_spec,
            volume_mount=self.volume_mount,
            node_selector=self.node_selector,
            image_pull_policy=self.image_pull_policy,
            resource_memory=self.resource_memory,
            resource_cpu=self.resource_cpu,
            experiment=self.experiment,
            alert_status=self.alert_status
        )



# 定义model
from  myapp.models.model_job import Crd
class Experiments(Model,Crd,MyappModelBase):
    __tablename__ = 'experiments'

    @property
    def url(self):
        if self.status=='' or self.status=='Created':
            katib_url = conf.get('KATIB_URL') + "/katib/hp_monitor/"
            return Markup(f'<a target=_blank href="{katib_url}">{self.name}</a>')
        else:
            katib_url = conf.get('KATIB_URL')+ "/katib/hp_monitor/%s/%s"%(self.namespace,self.name)
            return Markup(f'<a target=_blank href="{katib_url}">{self.name}</a>')


