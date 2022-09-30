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


class Dimension_table(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'dimension'
    __table_args__ = (UniqueConstraint('sqllchemy_uri', 'table_name'),)
    id = Column(Integer, primary_key=True)
    sqllchemy_uri = Column(String(255),nullable=True)
    table_name = Column(String(255),nullable=True,unique=True)
    label = Column(String(255), nullable=True)
    describe = Column(String(2000), nullable=True)
    app = Column(String(255), nullable=True)
    owner = Column(String(2000), nullable=True,default='')
    columns=Column(Text, nullable=True,default='{}')
    status = Column(Integer, default=1)


    @property
    def table_html(self):
        users=''
        users = users+self.owner if self.owner else users
        users = users.split(',')
        users = [user.strip() for user in users if user.strip()]
        url_path = conf.get('MODEL_URLS',{}).get("dimension")
        if g.user.is_admin() or g.user.username in users or '*' in self.owner:
            return Markup(f'<a target=_blank href="{url_path}?targetId={self.id}">{self.table_name}</a>')
        else:
            return self.table_name

    @property
    def operate_html(self):
        url=f'''
        <a target=_blank href="/dimension_table_modelview/api/create_external_table/%s">创建远程表</a>  | <a target=_blank href="/dimension_table_modelview/api/external/%s">建外表示例</a> | <a href="/dimension_table_modelview/api/clear/%s">清空表记录</a>
        '''%(self.id,self.id,self.id)
        return Markup(url)





