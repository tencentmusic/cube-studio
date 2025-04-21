from flask_appbuilder import Model
from flask import Markup
from sqlalchemy import (
    Boolean,
    Text,
)
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app
from myapp.models.helpers import ImportMixin
from sqlalchemy import Column, Integer, String

from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Metadata_metric(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'metadata_metric'
    id = Column(Integer, primary_key=True,comment='id主键')
    app = Column(String(100), nullable=False,comment='应用名')
    name = Column(String(300),nullable=True,comment='英文名')
    label = Column(String(300), nullable=True,comment='中文名')
    describe = Column(String(500),nullable=False,comment='描述')
    caliber = Column(Text(65536), nullable=True,default='',comment='数据负责人')
    metric_type = Column(String(100),nullable=True,comment='指标类型  	原子指标  衍生指标')    #
    metric_level = Column(String(100), nullable=True,default= __('普通'),comment='指标等级   普通  重要  核心')    #
    metric_dim = Column(String(100), nullable=True, default='',comment='指标维度   天  月   周')  #
    metric_data_type = Column(String(100), nullable=True, default='',comment='指标类型  营收/规模/商业化')  #
    metric_responsible = Column(String(200), nullable=True, default='',comment='指标负责人')  #
    status = Column(String(100), nullable=True, default='',comment='状态  下线  上线   创建中')  #
    task_id = Column(String(200), nullable=True, default='',comment='所有相关任务id')  #
    public = Column(Boolean, default=True,comment='是否公开')  #
    remark = Column(Text(65536), nullable=True,default='',comment='备注')   #
    expand = Column(Text(65536), nullable=True,default='{}',comment='扩展参数')
    def __repr__(self):
        return self.name

    @property
    def remark_html(self):
        return "<br>".join(self.remark.split('\n'))

    def clone(self):
        return Metadata_metric(
            app=self.app,
            name=self.name,
            describe=self.describe,
            caliber=self.caliber,
            metric_type=self.metric_type,
            metric_level=self.metric_level,
            metric_dim=self.metric_dim,
            metric_data_type=self.metric_data_type,
            metric_responsible=self.metric_responsible,
            status=self.status,
            task_id=self.task_id,
            expand=self.expand,
        )
