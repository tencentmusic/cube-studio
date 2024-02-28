from flask_appbuilder import Model
from sqlalchemy import Float
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Text,
    Enum,
)
from myapp.models.base import MyappModelBase
from myapp.models.model_team import Project
from myapp.models.helpers import AuditMixinNullable
from flask import request
from myapp import app
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from sqlalchemy import Column, Integer, String, ForeignKey
from flask_appbuilder.models.decorators import renders
from flask import Markup
metadata = Model.metadata
conf = app.config



class NNI(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'nni'
    id = Column(Integer, primary_key=True,comment='id主键')
    job_type = Column(Enum('Job',name='job_type'),nullable=True,default='Job',comment='任务类型')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False,comment='项目组id')  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    name = Column(String(200), unique = True, nullable=False,comment='英文名')
    namespace = Column(String(200), nullable=True,default='automl',comment='命名空间')
    describe = Column(Text,comment='描述')
    parallel_trial_count = Column(Integer,default=3,comment='最大并行数')
    maxExecDuration = Column(Integer,default=3600,comment='最大执行时长')
    max_trial_count = Column(Integer,default=12,comment='最大搜索次数')
    max_failed_trial_count = Column(Integer,default=3,comment='最大失败次数')
    objective_type = Column(Enum('maximize','minimize',name='objective_type'),nullable=False,default='maximize',comment='搜索目标类型')
    objective_goal = Column(Float, nullable=False,default=0.99,comment='搜索目标值')
    objective_metric_name = Column(String(200), nullable=False,default='accuracy',comment='目标度量名称')
    objective_additional_metric_names = Column(String(200),default='',comment='附加目标度量名')   # 逗号分隔
    algorithm_name = Column(String(200),nullable=False,default='Random',comment='算法名')
    algorithm_setting = Column(Text,default='',comment='算法配置')  # 搜索算法的配置
    parameters=Column(Text,default='{}',comment='搜索超参的配置')  #
    job_json = Column(Text, default='{}',comment='根据不同算法和参数写入的task模板')  #
    trial_spec=Column(Text,default='',comment='根据不同算法和参数写入的task模板')    #
    # code_dir = Column(String(200), default='')  # 代码挂载
    job_worker_image = Column(String(200),nullable=True,default='',comment='执行镜像')
    job_worker_command = Column(String(200), nullable=True, default='',comment='执行命令')
    working_dir = Column(String(200), default='',comment='启动目录')  # 挂载
    volume_mount = Column(String(2000), default='kubeflow-user-workspace(pvc):/mnt',comment='挂载')  # 挂载
    node_selector = Column(String(100), default='cpu=true,train=true',comment='机器选择器')  # 挂载
    image_pull_policy = Column(Enum('Always', 'IfNotPresent',name='image_pull_policy'), nullable=False, default='Always',comment='镜像拉取策略')
    resource_memory = Column(String(100), default='1G',comment='申请内存')
    resource_cpu = Column(String(100), default='1',comment='申请cpu')
    resource_gpu = Column(String(100), default='0',comment='申请gpu')
    experiment=Column(Text,default='',comment='构建出来的实验体')  #
    alert_status = Column(String(100), default='Pending,Running,Succeeded,Failed,Terminated',comment='哪些状态会报警Pending,Running,Succeeded,Failed,Unknown,Waiting,Terminated')   #



    def __repr__(self):
        return self.name

    @property
    def run(self):
        ops_html = f'<a target=_blank href="/nni_modelview/run/{self.id}">{__("运行")}</a> | <a target=_blank href="/k8s/web/search/{self.project.cluster["NAME"]}/{conf.get("AUTOML_NAMESPACE")}/{self.name}">{__("容器")}</a>  | <a href="/nni_modelview/stop/{self.id}">{__("清理")}</a> '
        return Markup(ops_html)

    # @property
    # def parameters_html(self):
    #     return Markup('<pre><code>' + self.parameters + '</code></pre>')


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
        return Markup(f'<a target=_blank href="/nni_modelview/api/web/{self.id}">{self.describe}</a>')


    # @property
    # def trial_spec_html(self):
    #     return Markup('<pre><code>' + self.trial_spec + '</code></pre>')


    # @property
    # def experiment_html(self):
    #     return Markup('<pre><code>' + self.experiment + '</code></pre>')

    @property
    def log(self):
        return Markup(f'<a target=_blank href="/nni_modelview/log/{self.id}">log</a>')


    def get_node_selector(self):
        return self.get_default_node_selector(self.project.node_selector,self.resource_gpu,'train')



    def clone(self):
        return NNI(
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


