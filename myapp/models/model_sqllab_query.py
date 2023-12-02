from flask_appbuilder import Model
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from myapp import app
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime,Text
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config
import pysnooper

# 定义model
class Sqllab_Query(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'sqlab_query'
    id = Column(Integer, primary_key=True)
    submit_time = Column(String(40), nullable=False, default='')
    start_time = Column(String(40), nullable=False, default='')
    end_time = Column(String(40), nullable=False, default='')

    engine_arg1 = Column(String(200), nullable=False, default='')
    engine_arg2 = Column(String(200), nullable=False, default='')

    qsql = Column(String(5000), nullable=False, default='')
    engine = Column(String(100), nullable=False, default='mysql')
    deli =  Column(String(11), nullable=False, default='|')

    stage =  Column(String(11), nullable=False, default='START')
    status =  Column(String(11), nullable=False, default='INIT')

    task_id = Column(String(100), nullable=False, default='')

    log_url = Column(String(400), nullable=False, default='')
    ui_url = Column(String(400), nullable=False, default='')
    result_url = Column(String(400), nullable=False, default='')
    result_line_num = Column(String(400), nullable=False, default='-1')
    err_msg = Column(Text)

    username = Column(String(400), nullable=False, default='')


