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
from myapp.utils.py.py_k8s import K8s
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from .model_team import Project
from .model_job import Pipeline
from myapp import app,db
from myapp.models.base import MyappModelBase
from myapp.models.helpers import ImportMixin
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
import datetime
metadata = Model.metadata
conf = app.config


# 定义训练 model
class Training_Model(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'model'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(100))
    describe = Column(String(1000))
    path = Column(String(200))
    download_url = Column(String(200))
    project_id = Column(Integer, ForeignKey('project.id'))  # 定义外键
    project = relationship(
        Project, foreign_keys=[project_id]
    )
    pipeline_id = Column(Integer,default=0)  # 定义外键
    run_id = Column(String(100),nullable=False)   # 可能同一个pipeline产生多个模型
    run_time = Column(String(100))
    framework = Column(String(100))
    metrics = Column(Text,default='{}')
    md5 = Column(String(200),default='')
    api_type = Column(String(100))

    def __repr__(self):
        return self.name

    @property
    def pipeline_url(self):
        if self.pipeline_id:
            pipeline = db.session.query(Pipeline).filter_by(id=self.pipeline_id).first()
            if pipeline:
                return Markup(f'<a target=_blank href="/frontend/showOutLink?url=%2Fstatic%2Fappbuilder%2Fvison%2Findex.html%3Fpipeline_id%3D{self.pipeline_id}">{pipeline.describe}</a>')

        return Markup(f'未知')

    @property
    def project_url(self):
        if self.project:
            return Markup(f'{self.project.name}({self.project.describe})')
        elif self.pipeline and self.pipeline.project:
            return Markup(f'{self.pipeline.project.name}({self.pipeline.project.describe})')
        else:
            return Markup(f'未知')

    @property
    def deploy(self):
        ops=f'''
        <a href="/training_model_modelview/deploy/{self.id}">发布</a> 
        '''
        return Markup(ops)

