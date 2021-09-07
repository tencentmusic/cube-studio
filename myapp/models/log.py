"""A collection of ORM sqlalchemy models for Myapp"""
from contextlib import closing
from copy import copy, deepcopy
from datetime import datetime
from flask_appbuilder import Model
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
)
from sqlalchemy.engine import url
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import relationship, sessionmaker, subqueryload
from myapp import (
    app,
    appbuilder,
    conf,
    db
)
from myapp.models.base import MyappModelBase
from myapp import app, db, is_feature_enabled, security_manager
from myapp.security import MyUser


config = app.config
custom_password_store = config.get("SQLALCHEMY_CUSTOM_PASSWORD_STORE")
stats_logger = config.get("STATS_LOGGER")

PASSWORD_MASK = "X" * 10


class Log(Model,MyappModelBase):

    """ORM object used to log Myapp actions to the database"""

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("ab_user.id"))
    user = relationship(
        MyUser, foreign_keys=[user_id]
    )
    action = Column(String(512))
    method = Column(String(50))
    path = Column(String(200))
    status = Column(Integer)
    json = Column(Text)
    dttm = Column(DateTime, default=datetime.now)  # 不要使用datetime.now()不然是程序启动的固定时间了
    duration_ms = Column(Integer)
    referrer = Column(String(1024))


    label_columns={
        'user':'用户',
        "action": "函数",
        "method": "方法",
        "path": "网址",
        "status": "状态",
        "dttm": "时间",
        "duration_ms": "响应延迟",
        "referrer": "相关人",
    }


