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
from sqlalchemy import String,Column,Integer,ForeignKey,UniqueConstraint,BigInteger,TIMESTAMP
import numpy
import random
import copy
import logging
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from flask import escape, g, Markup, request
from .model_team import Project
from myapp import app,db
from myapp.models.helpers import ImportMixin
# from myapp.models.base import MyappModel
# 添加自定义model
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config
from myapp.utils import core
import re
from myapp.utils.py import py_k8s
import pysnooper

# 定义model
class Metadata_metric(Model,AuditMixinNullable,ImportMixin,MyappModelBase):
    __tablename__ = 'metadata_metric'
    id = Column(Integer, primary_key=True)
    app = Column(String(100), nullable=False)
    name = Column(String(300),nullable=True)
    label = Column(String(300), nullable=True)
    describe = Column(String(500),nullable=False)
    caliber = Column(Text(65536), nullable=True,default='')
    metric_type = Column(String(100),nullable=True)    # 指标类型  	原子指标  衍生指标
    metric_level = Column(String(100), nullable=True,default='普通')    # 指标等级   普通  重要  核心
    metric_dim = Column(String(100), nullable=True, default='')  # 指标维度   天  月   周
    metric_data_type = Column(String(100), nullable=True, default='')  # 指标类型  营收/规模/商业化
    metric_responsible = Column(String(200), nullable=True, default='')  # 指标负责人
    status = Column(String(100), nullable=True, default='')  # 状态  下线  上线   创建中
    task_id = Column(String(200), nullable=True, default='')  # 所有相关任务id
    public = Column(Boolean, default=True)  # 是否公开
    expand = Column(Text(65536), nullable=True,default='{}')
    def __repr__(self):
        return self.name


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
