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
class Sqllab_Query(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'idex_query'
#    __table_args__ = (UniqueConstraint('db', 'table'),)
    id = Column(Integer, primary_key=True)
    submit_time = Column(String(40), nullable=False, default='')
    start_time = Column(String(40), nullable=False, default='')
    end_time = Column(String(40), nullable=False, default='')
    group_id = Column(String(100), nullable=False, default='')
    qsql = Column(String(5000), nullable=False, default='')
    engine = Column(String(100), nullable=False, default='thive')
    deli =  Column(String(11), nullable=False, default='|')

    stage =  Column(String(11), nullable=False, default='START')
    status =  Column(String(11), nullable=False, default='INIT')

    gaia_id = Column(String(100), nullable=False, default='')

    task_id = Column(String(100), nullable=False, default='')

    spark_log_url = Column(String(400), nullable=False, default='')
    spark_ui_url = Column(String(400), nullable=False, default='')
    result_url = Column(String(400), nullable=False, default='')
    result_line_num = Column(String(400), nullable=False, default='-1')
    err_msg = Column(String(5000), nullable=False, default='')

    rtx = Column(String(400), nullable=False, default='')
    biz = Column(String(400), nullable=False, default='')

#    @property
#    def operator(self):
#        if self.is_central_task==1:
#            return "提交"
#        return Markup(f'<a target=_blank href="/data_access_offline_modelview/api/active/{self.id}">提交</a>')


