import traceback

from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder import ModelView, ModelRestApi
from flask_appbuilder import ModelView,AppBuilder,expose,BaseView,has_access
from importlib import reload
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_babel import lazy_gettext,gettext
from flask_appbuilder.forms import GeneralModelConverter
import uuid
from sqlalchemy import and_, or_, select
from flask_appbuilder.actions import action
import re,os
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from kfp import compiler
from sqlalchemy.exc import InvalidRequestError
from myapp import app, appbuilder,db,event_logger
from myapp.utils import core
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp.views.view_dimension import Dimension_table_ModelView_Api


from .baseApi import (
    MyappModelRestApi
)
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from myapp import security_manager
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
from myapp.models.model_metadata_metric import Metadata_metric
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.security import MyUser
conf = app.config
logging = app.logger


class Metadata_Metrics_table_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query
        # 公开的，或者责任人包含自己的
        return query.filter(
            or_(
                self.model.public==True,
                self.model.metric_responsible.contains(g.user.username)
            )
        )



class Metadata_metric_ModelView_base():
    label_title='指标'
    datamodel = SQLAInterface(Metadata_metric)
    base_permissions = ['can_add','can_show','can_edit','can_list','can_delete']
    base_order = ("id", "desc")
    order_columns = ['id']
    search_columns=['metric_data_type','metric_responsible','app','name','label','describe','metric_type','metric_level','task_id','caliber']
    show_columns=['id','app','metric_data_type','name','label','describe','metric_type','metric_level','metric_dim','metric_responsible','caliber','task_id','public']
    list_columns = ['app','metric_data_type','name','label','describe','metric_level','metric_responsible','public','metric_type','task_id']
    cols_width = {
        "name":{"type": "ellip2", "width": 200},
        "label": {"type": "ellip2", "width": 200},
        "describe": {"type": "ellip2", "width": 400},
        "metric_responsible": {"type": "ellip2", "width": 300}
    }
    spec_label_columns={
        "name":"指标英文名",
        "label": "指标中文名",
        "describe":"指标描述",
        "metric_data_type":"指标模块",
        "task_id":"任务id"
    }
    add_columns = ['app','metric_data_type','name','label','describe','metric_type','metric_level','metric_dim','metric_responsible','caliber','task_id','public']
    # show_columns = ['project','name','describe','config_html','dag_json_html','created_by','changed_by','created_on','changed_on','expand_html']
    edit_columns = add_columns
    base_filters = [["id", Metadata_Metrics_table_Filter, lambda: []]]
    add_form_extra_fields = {
        "app": SelectField(
            label=_(datamodel.obj.lab('app')),
            description='产品',
            widget=MySelect2Widget(can_input=True,conten2choices=True),
            choices=[[x,x] for x in ['产品1',"产品2","产品3"]]
        ),
        "name":StringField(
            label=_(datamodel.obj.lab('name')),
            description='指标英文名',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "label": StringField(
            label=_(datamodel.obj.lab('label')),
            description='指标中文名',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "describe": StringField(
            label=_(datamodel.obj.lab('describe')),
            description='指标描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "metric_type": SelectField(
            label=_(datamodel.obj.lab('metric_type')),
            description='指标类型',
            default='原子指标',
            widget=Select2Widget(),
            choices=[[x,x] for x in ['原子指标', '衍生指标']]
        ),
        "metric_data_type": SelectField(
            label=_(datamodel.obj.lab('metric_data_type')),
            description='指标归属模块',
            widget=MySelect2Widget(can_input=True,conten2choices=True),
            choices=[[x,x] for x in ['模块1',"模块2","模块3"]]
        ),
        "metric_level":SelectField(
            label=_(datamodel.obj.lab('metric_level')),
            description='指标重要级别',
            default='普通',
            widget=Select2Widget(),
            choices=[[x,x] for x in ['普通','重要','核心']]
        ),
        "metric_dim": SelectField(
            label=_(datamodel.obj.lab('metric_dim')),
            description='指标维度1',
            default='天',
            widget=Select2Widget(),
            choices=[[x, x] for x in ['天', '周', '月']]
        ),
        "metric_responsible": StringField(
            label=_(datamodel.obj.lab('metric_responsible')),
            description='指标负责人，逗号分隔',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "status": SelectField(
            label=_(datamodel.obj.lab('status')),
            description='指标状态',
            widget=Select2Widget(),
            choices=[[x,x] for x in [ "下线","待审批",'创建中',"上线",]]
        ),
        "caliber": StringField(
            label=_(datamodel.obj.lab('caliber')),
            description='指标口径描述，代码和计算公式',
            widget=MyBS3TextAreaFieldWidget(rows=3),
            validators=[DataRequired()]
        ),
        "task_id": StringField(
            label=_(datamodel.obj.lab('task_id')),
            description='任务id',
            widget=BS3TextFieldWidget()
        ),
    }

    edit_form_extra_fields = add_form_extra_fields
    import_data=True
    download_data=True


    def pre_upload(self,data):
        data['public'] = bool(int(data.get('public', 1)))
        return data

    @action(
        "muldelete", __("Delete"), __("Delete all Really?"), "fa-trash", single=False
    )
    def muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            try:
                if item.created_by.username==g.user.username:
                    self.pre_delete(item)
                    self.datamodel.delete(item, raise_exception=True)
                    self.post_delete(item)
            except Exception as e:
                flash(str(e), "danger")

class Metadata_metric_ModelView_Api(Metadata_metric_ModelView_base,MyappModelRestApi):
    datamodel = SQLAInterface(Metadata_metric)
    route_base = '/metadata_metric_modelview/api'
    label_title='指标'


appbuilder.add_api(Metadata_metric_ModelView_Api)








