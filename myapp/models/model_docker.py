from flask_appbuilder import Model
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Boolean,
    Text
)
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.models.base import MyappModelBase
from myapp.models.model_team import Project
from myapp.models.helpers import AuditMixinNullable
from myapp import app
from sqlalchemy import Column, Integer, String, ForeignKey

from flask import Markup
metadata = Model.metadata
conf = app.config


class Docker(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'docker'
    id = Column(Integer, primary_key=True,comment='id主键')
    project_id = Column(Integer, ForeignKey('project.id'), nullable=False,comment='项目组id')
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    describe = Column(String(200),  nullable=True,comment='描述')
    base_image = Column(String(200),  nullable=True,comment='基础镜像')
    target_image=Column(String(200), nullable=True,default='',comment='目标镜像')
    last_image = Column(String(200), nullable=True, default='',comment='最后生成镜像')
    need_gpu = Column(Boolean, nullable=True, default=False,comment='是否需要gpu')
    consecutive_build = Column(Boolean, default=True,comment='连续构建')
    expand = Column(Text(65536), default='{}',comment='扩展参数')

    def __repr__(self):
        return self.describe

    @property
    def save(self):
        return Markup(f'<a href="/docker_modelview/api/save/{self.id}">{__("保存")}</a>')

    @property
    def debug(self):
        return Markup(f'<a target=_blank href="/docker_modelview/api/debug/{self.id}">{__("调试")}</a> | <a  href="/docker_modelview/api/delete_pod/{self.id}">{__("清理")}</a> | <a target=_blank href="/docker_modelview/api/save/{self.id}">{__("保存")}</a>')

    @property
    def image_history(self):
        return Markup(f'{__("基础镜像")}:{self.base_image}<br>{__("当前镜像")}:{self.last_image if self.last_image else self.base_image}<br>{__("目标镜像")}:{self.target_image}')
