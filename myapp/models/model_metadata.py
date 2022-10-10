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




class Metadata_table(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'metadata_table'
    id = Column(Integer, primary_key=True)
    node_id = Column(String(200))
    app = Column(String(200), nullable=True)
    c_org_fullname=Column(String(255), nullable=True, default='')
    db = Column(String(200), nullable=True)  # 库名
    table = Column(String(400),nullable=True)
    metadata_column = Column(Text, nullable=True,default='[]')
    describe = Column(String(200), nullable=True)
    owner = Column(String(200),nullable=True)
    lifecycle=Column(Integer, nullable=True, default=0)
    rec_lifecycle = Column(Integer, nullable=True, default=0)
    storage_size = Column(String(200), nullable=True,default='0')
    storage_cost = Column(Float, nullable=True, default=0)
    visits_seven = Column(Integer, nullable=True, default=0)
    visits_thirty = Column(BigInteger, nullable=True, default=0)
    visits_sixty = Column(BigInteger, nullable=True, default=0)
    recent_visit = Column(Date, nullable=True)
    partition_start = Column(String(255), nullable=True, default='')
    partition_end = Column(String(255), nullable=True, default='')
    status = Column(String(255), nullable=True, default='')
    creator = Column(String(255), nullable=True, default='')

    create_table_ddl = Column(Text, nullable=True)
    col_info = Column(Text, nullable=True)
    partition_update_mode = Column(Integer, nullable=True)
    is_privilege = Column(Integer, nullable=False,default=1)
    data_source = Column(Integer, nullable=True,default=0)

    field = Column(String(200), nullable=True,default='未知')  # 所属数据域
    security_level = Column(String(200), nullable=True,default='普通')  # 安全等级
    value_score = Column(String(200), nullable=True,default='0')   # 价值评估
    warehouse_level = Column(String(200), nullable=True,default='未知')   # 数仓级别
    ttl = Column(String(200), nullable=True)   # 保留时长

    expand = Column(Text(65536), nullable=True,default='{}')

    def __repr__(self):
        return self.table


