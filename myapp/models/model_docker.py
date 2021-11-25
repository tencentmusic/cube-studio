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
import pysnooper
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
# from myapp.utils.py.py_k8s import K8s


# 定义model
class Docker(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'docker'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    describe = Column(String(200),  nullable=True)
    base_image = Column(String(200),  nullable=True)
    target_image=Column(String(200), nullable=True,default='')
    last_image = Column(String(200), nullable=True, default='')
    need_gpu = Column(Boolean, nullable=True, default=False)
    consecutive_build = Column(Boolean, default=True)  # 连续构建
    expand = Column(Text(65536), default='{}')

    def __repr__(self):
        return self.label

    # 清空激活
    @property
    def save(self):
        return Markup(f'<a href="/docker_modelview/save/{self.id}">保存</a>')

    # 清空激活
    @property
    def debug(self):
        return Markup(f'<a href="/docker_modelview/debug/{self.id}">调试</a> | <a href="/docker_modelview/delete_pod/{self.id}">清理</a>')

    @property
    def image_history(self):
        return Markup(f'基础镜像:{self.base_image}<br>当前镜像:{self.last_image if self.last_image else self.base_image}<br>目标镜像:{self.target_image}')
