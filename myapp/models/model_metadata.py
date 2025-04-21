import datetime

from flask_appbuilder import Model
from sqlalchemy import Float
from sqlalchemy import Text
from sqlalchemy import BigInteger
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app
from myapp.models.helpers import ImportMixin

from sqlalchemy import Column, Integer, String, Date
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config




class Metadata_table(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'metadata_table'
    id = Column(Integer, primary_key=True,comment='id主键')
    node_id = Column(String(200),comment='唯一性id')
    app = Column(String(200), nullable=True,comment='所属应用')
    c_org_fullname=Column(String(255), nullable=True, default='',comment='所属组织架构')
    db = Column(String(200), nullable=True,comment='库名')  #
    table = Column(String(400),nullable=True,comment='表名')
    metadata_column = Column(Text, nullable=True,default='[]',comment='列信息')
    describe = Column(String(200), nullable=True,comment='描述')
    owner = Column(String(200),nullable=True,comment='责任人')
    lifecycle=Column(Integer, nullable=True, default=0,comment='生命周期')
    rec_lifecycle = Column(Integer, nullable=True, default=0,comment='推荐生命周期')
    storage_size = Column(String(200), nullable=True,default='0',comment='存储大小')
    storage_cost = Column(Float, nullable=True, default=0,comment='存储成本')
    visits_seven = Column(Integer, nullable=True, default=0,comment='7日访问量')
    visits_thirty = Column(BigInteger, nullable=True, default=0,comment='30日访问量')
    visits_sixty = Column(BigInteger, nullable=True, default=0,comment='60日访问量')
    recent_visit = Column(Date, nullable=True,comment='最新被访问时间')
    partition_start = Column(String(255), nullable=True, default='',comment='分区起点')
    partition_end = Column(String(255), nullable=True, default='',comment='分区终点')
    status = Column(String(255), nullable=True, default='',comment='状态')
    creator = Column(String(255), nullable=True, default='',comment='创建者')

    create_table_ddl = Column(Text, nullable=True,comment='建表语句')
    col_info = Column(Text, nullable=True,comment='列信息')
    partition_update_mode = Column(Integer, nullable=True,comment='分区更新模式')
    is_privilege = Column(Integer, nullable=False,default=1,comment='是否特权')
    data_source = Column(Integer, nullable=True,default=0,comment='数据来源')

    field = Column(String(200), nullable=True,default= 'unknown',comment='所属数据域')  #
    security_level = Column(String(200), nullable=True,default= __('普通'),comment='安全等级')  #
    value_score = Column(String(200), nullable=True,default='0',comment='价值评估')   #
    warehouse_level = Column(String(200), nullable=True,default= 'unknown',comment='数仓级别')   #
    ttl = Column(String(200), nullable=True,comment='保留时长')   #

    expand = Column(Text(65536), nullable=True,default='{}',comment='扩展参数')

    def __repr__(self):
        return self.table


