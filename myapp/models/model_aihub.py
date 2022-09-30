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

class Aihub(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'aihub'

    id = Column(Integer, primary_key=True)
    uuid = Column(String(2000), nullable=True, default='')
    status = Column(String(200), nullable=True, default='')
    doc = Column(String(200), nullable=True,default='')
    name = Column(String(200), nullable=True, default='')
    field = Column(String(200), nullable=True, default='')
    scenes = Column(String(200), nullable=True, default='')
    type = Column(String(200), nullable=True, default='')
    label = Column(String(200), nullable=True, default='')
    describe = Column(String(2000), nullable=True, default='')
    source = Column(String(200), nullable=True, default='')
    pic = Column(String(500), nullable=True, default='')
    dataset = Column(Text, nullable=True, default='')
    notebook = Column(String(2000), nullable=True, default='')
    job_template = Column(Text, nullable=True, default='')
    pre_train_model = Column(String(2000), nullable=True, default='')
    inference = Column(Text, nullable=True, default='')
    service = Column(Text, nullable=True, default='')
    version = Column(String(200), nullable=True, default='')
    hot = Column(Integer, default=1)
    price = Column(Integer, default=1)


    @property
    def card(self):
        return Markup(f'''
<div>
<a href="{self.doc}"> <img src="{self.pic}" style="border: black;border-radius: 5px; border-width: 2px;padding: 10px" height=220px width=360px border=5px alt="{self.describe}"/></a><br>
<div style="padding-top:3px;">
<p class="ellip1">{self.describe}</p>
<a style="padding-left:1px;padding-right:6px;" href='https://www.baidu.com'><button> 调试 </button></a>
<a style="padding-left:6px;padding-right:6px;" href='https://www.baidu.com'><button> 训练 </button></a>
<a style="padding-left:6px;padding-right:1px;" href='https://www.baidu.com'><button> 服务 </button></a>
</div>
</div>
''')

    @property
    def train_html(self):
        return '训练'

    @property
    def notebook_html(self):
        return 'notebook打开'

    @property
    def service_html(self):
        return '打开web体验'




