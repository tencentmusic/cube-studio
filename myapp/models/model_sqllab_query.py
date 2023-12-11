from flask_appbuilder import Model
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from myapp import app
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime,Text
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config
import pysnooper
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _

# 定义model
class Sqllab_Query(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'sqlab_query'
    id = Column(Integer, primary_key=True,comment='id主键')
    submit_time = Column(String(40), nullable=False, default='',comment='提交时间')
    start_time = Column(String(40), nullable=False, default='',comment='启动时间')
    end_time = Column(String(40), nullable=False, default='',comment='结束时间')

    engine_arg1 = Column(String(200), nullable=False, default='',comment='引擎参数1')
    engine_arg2 = Column(String(200), nullable=False, default='',comment='引擎参数2')

    qsql = Column(String(5000), nullable=False, default='',comment='sql语句')
    engine = Column(String(100), nullable=False, default='mysql',comment='引擎')
    deli =  Column(String(11), nullable=False, default='|',comment='分隔符')

    stage =  Column(String(11), nullable=False, default='START',comment='阶段')
    status =  Column(String(11), nullable=False, default='INIT',comment='状态')

    task_id = Column(String(100), nullable=False, default='',comment='任务id')

    log_url = Column(String(400), nullable=False, default='',comment='日志url')
    ui_url = Column(String(400), nullable=False, default='',comment='任务url')
    result_url = Column(String(400), nullable=False, default='',comment='结果url')
    result_line_num = Column(String(400), nullable=False, default='-1',comment='结果行数')
    err_msg = Column(Text,comment='报错消息')

    username = Column(String(400), nullable=False, default='',comment='用户名')


