from flask_appbuilder import Model
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Boolean,
    Text
)
from myapp.models.base import MyappModelBase
from myapp.models.model_team import Project
from myapp.models.helpers import AuditMixinNullable
from myapp import app
from sqlalchemy import Column, Integer, String, ForeignKey

from flask import Markup
metadata = Model.metadata
conf = app.config
# from myapp.utils.py.py_k8s import K8s


class Docker(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'docker'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    describe = Column(String(200),  nullable=True)
    base_image = Column(String(200),  nullable=True)
    target_image=Column(String(200), nullable=True,default='')
    last_image = Column(String(200), nullable=True, default='')
    need_gpu = Column(Boolean, nullable=True, default=False)
    consecutive_build = Column(Boolean, default=True)
    expand = Column(Text(65536), default='{}')

    def __repr__(self):
        return self.label

    @property
    def save(self):
        return Markup(f'<a href="/docker_modelview/save/{self.id}">保存</a>')

    @property
    def debug(self):
        return Markup(f'<a target=_blank href="/docker_modelview/debug/{self.id}">调试</a> | <a  href="/docker_modelview/delete_pod/{self.id}">清理</a> | <a target=_blank href="/docker_modelview/save/{self.id}">保存</a>')

    @property
    def image_history(self):
        return Markup(f'基础镜像:{self.base_image}<br>当前镜像:{self.last_image if self.last_image else self.base_image}<br>目标镜像:{self.target_image}')
