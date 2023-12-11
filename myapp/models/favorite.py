from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey,Float
from sqlalchemy.orm import relationship
import datetime,time,json
from myapp import app,db

from myapp.models.helpers import AuditMixinNullable, ImportMixin
from sqlalchemy.orm import backref, relationship
from myapp.models.base import MyappModelBase
from myapp.models.helpers import ImportMixin

from sqlalchemy import String,Column,Integer,ForeignKey,UniqueConstraint
from myapp.security import MyUser
metadata = Model.metadata
conf = app.config
import pysnooper


class Favorite(Model,MyappModelBase):
    __tablename__ = "favorite"
    id = Column(Integer, primary_key=True,comment='id主键')
    model_name = Column(String(100), nullable=False,comment='数据结构')
    row_id = Column(String(500), nullable=False,comment='数据id')
    user_id = Column(Integer,comment='用户id')

    def __repr__(self):
        return "userid(%s)model(%s)rowid(%s)"%(self.user_id,self.model_name,self.row_id)

    __table_args__ = (
        UniqueConstraint('model_name','row_id','user_id'),
    )

