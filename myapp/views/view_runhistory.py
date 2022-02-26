from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface

# 将model添加成视图，并控制在前端的显示
from myapp.models.model_job import Repository,Images,Job_Template,Task,Pipeline,Workflow,Tfjob,Xgbjob,RunHistory,Pytorchjob

from myapp import app, appbuilder,db,event_logger
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget,BS3TextAreaFieldWidget
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from sqlalchemy import and_, or_, select

from .baseApi import (
    MyappModelRestApi
)

from myapp import security_manager
import kfp    # 使用自定义的就要把pip安装的删除了
from werkzeug.datastructures import FileStorage
from .base import (
    api,
    BaseMyappView,
    check_ownership,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,
)
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
logging = app.logger


class RunHistory_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        pipeline_ids = security_manager.get_create_pipeline_ids(db.session)
        return query.filter(
            or_(
                self.model.pipeline_id.in_(pipeline_ids),
                # self.model.project.name.in_(['public'])
            )
        )



class RunHistory_ModelView_Base():
    label_title='定时调度历史'
    datamodel = SQLAInterface(RunHistory)
    base_order = ('id', 'desc')
    order_columns = ['id']

    list_columns = ['pipeline_url','creator','created_on','execution_date','status_url','log','history']
    edit_columns = ['status']
    base_filters = [["id", RunHistory_Filter, lambda: []]]  # 设置权限过滤器
    add_form_extra_fields = {
        "status": SelectField(
            _(datamodel.obj.lab('status')),
            description="状态comed为已识别未提交，created为已提交",
            widget=Select2Widget(),
            choices=[['comed', 'comed'], ['created', 'created']]
        ),
    }
    edit_form_extra_fields = add_form_extra_fields


class RunHistory_ModelView(RunHistory_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(RunHistory)

appbuilder.add_view(RunHistory_ModelView,"定时调度记录",icon = 'fa-clock-o',category = '训练')




# 添加api
class RunHistory_ModelView_Api(RunHistory_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(RunHistory)
    route_base = '/runhistory_modelview/api'

appbuilder.add_api(RunHistory_ModelView_Api)



