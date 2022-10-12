from flask_appbuilder import Model
from sqlalchemy.orm import relationship
from sqlalchemy import Text

from myapp.models.helpers import AuditMixinNullable
from .model_team import Project
from .model_job import Pipeline
from myapp import app,db
from myapp.models.base import MyappModelBase
from sqlalchemy import Column, Integer, String, ForeignKey
from flask import Markup
metadata = Model.metadata
conf = app.config


class Training_Model(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'model'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(100))
    describe = Column(String(1000))
    path = Column(String(200))
    download_url = Column(String(200))
    project_id = Column(Integer, ForeignKey('project.id'))
    project = relationship(
        Project, foreign_keys=[project_id]
    )
    pipeline_id = Column(Integer,default=0)
    run_id = Column(String(100),nullable=False)   # pipeline run instance
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

        return Markup('未知')

    @property
    def project_url(self):
        if self.project:
            return Markup(f'{self.project.name}({self.project.describe})')
        elif self.pipeline and self.pipeline.project:
            return Markup(f'{self.pipeline.project.name}({self.pipeline.project.describe})')
        else:
            return Markup('未知')

    @property
    def deploy(self):
        ops=f'''
        <a href="/training_model_modelview/deploy/{self.id}">发布</a> 
        '''
        return Markup(ops)

