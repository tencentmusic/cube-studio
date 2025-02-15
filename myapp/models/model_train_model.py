import json
import os.path
import re
from flask_appbuilder import Model
from sqlalchemy.orm import relationship
from sqlalchemy import Text
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.helpers import AuditMixinNullable
from .model_team import Project
from .model_job import Pipeline
from myapp import app,db
from myapp.models.base import MyappModelBase
from sqlalchemy import Column, Integer, String, ForeignKey
from flask import Markup
metadata = Model.metadata
conf = app.config
import pysnooper

class Training_Model(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'model'
    id = Column(Integer, primary_key=True,comment='id主键')
    name = Column(String(100), nullable=False,comment='英文名')
    version = Column(String(100),comment='版本')
    describe = Column(String(1000),comment='描述')
    path = Column(String(200),comment='模型路径')
    download_url = Column(String(200),comment='下载url')
    project_id = Column(Integer, ForeignKey('project.id'),comment='项目组id')
    project = relationship(
        Project, foreign_keys=[project_id]
    )
    pipeline_id = Column(Integer,default=0,comment='任务流id')
    run_id = Column(String(100),nullable=False,comment='run id')   # pipeline run instance
    run_time = Column(String(100),comment='运行时间')
    framework = Column(String(100),comment='训练框架')
    metrics = Column(Text,default='{}',comment='指标')
    md5 = Column(String(200),default='',comment='md5值')
    api_type = Column(String(100),comment='api类型')
    expand = Column(Text(65536), default='{}',comment='扩展参数')

    def __repr__(self):
        return self.name

    @property
    def pipeline_url(self):
        if self.pipeline_id:
            pipeline = db.session.query(Pipeline).filter_by(id=self.pipeline_id).first()
            if pipeline:
                return Markup(f'<a target=_blank href="/frontend/showOutLink?url=%2Fstatic%2Fappbuilder%2Fvison%2Findex.html%3Fpipeline_id%3D{self.pipeline_id}">{pipeline.describe}</a>')

        return Markup('unknown')

    @property
    def project_url(self):
        if self.project:
            return Markup(f'{self.project.name}({self.project.describe})')
        else:
            return Markup('unknown')

    @property
    def deploy(self):
        download_url = f'{__("下载")} |'
        if self.download_url and self.download_url.strip():
            download_url = f'<a href="/training_model_modelview/api/download/{self.id}">{__("下载")}</a> |'
        if self.path and self.path.strip():
            if re.match('^/mnt/', self.path):
                local_path = f'/data/k8s/kubeflow/pipeline/workspace/{self.path.strip().replace("/mnt/","")}'
                if os.path.exists(local_path):
                    download_url = f'<a href="/training_model_modelview/api/download/{self.id}">{__("下载")}</a> |'
            if 'http://' in self.path or 'https://' in self.path:
                download_url = f'<a href="/training_model_modelview/api/download/{self.id}">{__("下载")}</a> |'

        ops=download_url+f'''
        <a href="/training_model_modelview/api/deploy/{self.id}">{__("发布")}</a> 
        '''
        return Markup(ops)

    @property
    def model_metric(self):
        try:
            metric_list = json.loads(self.metrics) if self.metrics else {}
            metric_str=''
            if type(metric_list)!=list:
                metric_list=[metric_list]
            for metric_json in metric_list:
                for metric_name in metric_json:
                    metric_str+=str(metric_name)+":"+str(metric_json[metric_name])+","
                metric_str=metric_str.strip(',')+'\n'
            return metric_str.strip()
        except Exception as e:
            print(e)
            return self.metrics


