from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import uuid
import urllib.parse
from sqlalchemy.exc import InvalidRequestError

from myapp.models.model_etl_pipeline import ETL_Pipeline,ETL_Task
from myapp.views.view_team import Project_Join_Filter
from flask_appbuilder.actions import action
from flask import jsonify
from flask_appbuilder.forms import GeneralModelConverter
from myapp.utils import core
from myapp import app, appbuilder,db
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from wtforms.validators import DataRequired, Length, Regexp
from sqlalchemy import or_
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget
import copy
from .baseApi import MyappModelRestApi
from flask import (
    flash,
    g,
    make_response,
    redirect,
    request,
)
from myapp import security_manager
from myapp.views.view_team import filter_join_org_project

from .baseFormApi import (
    MyappFormRestApi
)

from flask_appbuilder import expose
import datetime,time,json

conf = app.config
logging = app.logger

# todo： airflow的运行，删除，日志查询
class Airflow_ETL_PIPELINE():

    # @pysnooper.snoop(watch_explode=())
    # todo 任务流编排 运行按钮触发函数
    def run_pipeline(pipeline):
        # todo 检查任务是否存在，提交创建新的任务或修改旧任务，或者删除任务
        # todo 保存到调度平台，并发起远程调度
        pass

    # todo: 删除前先把下面的task删除了
    # @pysnooper.snoop()
    def delete_pipeline(self, pipeline):
        # 删除远程上下游关系
        # 删除本地
        exist_tasks = db.session.query(ETL_Task).filter_by(etl_pipeline_id=pipeline.id).all()
        for exist_task in exist_tasks:
            db.session.delete(exist_task)
            db.session.commit()

    # todo: pipeline的运行实例进度日志
    def log_pipeline(self,pipeline):
        pass

    # todo: 想要支持的模板列表
    # all_template = {}

    # todo: 任务删除
    def delete_task(self,task):
        pass

    # todo 任务日志
    def log_task(self,task):
        pass


