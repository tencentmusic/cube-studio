"""A collection of ORM sqlalchemy models for Myapp"""

from datetime import datetime
from flask_appbuilder import Model
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import relationship
from myapp.models.base import MyappModelBase
from myapp import app
from myapp.security import MyUser


config = app.config
custom_password_store = config.get("SQLALCHEMY_CUSTOM_PASSWORD_STORE")
stats_logger = config.get("STATS_LOGGER")

PASSWORD_MASK = "X" * 10


class Log(Model,MyappModelBase):

    """ORM object used to log Myapp actions to the database"""

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True,comment='id主键')

    user_id = Column(Integer, ForeignKey("ab_user.id"),comment='用户id')
    user = relationship(
        MyUser, foreign_keys=[user_id]
    )
    action = Column(String(512),comment='动作')
    method = Column(String(50),comment='方法')
    path = Column(String(200),comment='访问地址')
    status = Column(Integer,comment='状态')
    json = Column(Text,comment='请求提')
    dttm = Column(DateTime, default=datetime.now,comment='访问时间')  # 不要使用datetime.now()不然是程序启动的固定时间了
    duration_ms = Column(Integer,comment='持续时长')
    referrer = Column(String(1024),comment='来源地址')








