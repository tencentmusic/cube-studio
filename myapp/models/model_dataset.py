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
class Dataset(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'dataset'
    id = Column(Integer, primary_key=True)
    name =  Column(String(200), nullable=True)  #
    label = Column(String(200), nullable=True)  #
    describe = Column(String(2000), nullable=True) #

    source_type = Column(String(200), nullable=True)  # 数据集来源，开源，资产，采购
    source = Column(String(200), nullable=True)  # 数据集来源，github, 天池
    industry =  Column(String(200), nullable=True)  # 行业，
    icon = Column(String(2000), nullable=True)  # 图标
    field =  Column(String(200), nullable=True)  # 数据领域，视觉，听觉，文本
    usage =  Column(String(200), nullable=True)  # 数据用途
    research =  Column(String(200), nullable=True)  # 研究方向

    storage_class = Column(String(200), nullable=True, default='') # 存储类型，压缩
    file_type = Column(String(200), nullable=True,default='')  # 文件类型，图片 png，音频
    status = Column(String(200), nullable=True, default='')  # 文件类型  有待校验，已下线

    years = Column(String(200), nullable=True)  # 年份
    url  = Column(String(1000),nullable=True)  # 关联url
    path = Column(String(400),nullable=True)  # 本地的持久化路径
    download_url = Column(String(1000),nullable=True)  # 下载地址
    storage_size = Column(String(200), nullable=True,default='')  # 存储大小
    entries_num = Column(String(200), nullable=True, default='')  # 记录数目
    duration = Column(String(200), nullable=True, default='')  # 时长
    price = Column(String(200), nullable=True, default='0')  # 价格



    owner = Column(String(200),nullable=True)  #

    expand = Column(Text(65536), nullable=True,default='{}')

    def __repr__(self):
        return self.name

    @property
    def url_html(self):
        url='<a target=_blank href="%s">%s</a>'%(self.url,self.url)
        return Markup(url)

    @property
    def download_url_html(self):
        url='<a target=_blank href="%s">%s</a>'%(self.download_url,self.download_url)
        return Markup(url)
