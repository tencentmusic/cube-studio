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
from myapp.models.model_aihub import Aihub
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.security import MyUser
conf = app.config
logging = app.logger


# 获取某类project分组
class Aihub_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        # user_roles = [role.name.lower() for role in list(get_user_roles())]
        # if "admin" in user_roles:
        #     return query.filter(Project.type == value).order_by(Project.id.desc())
        return query.filter(self.model.field==value).order_by(self.model.id.desc())


class Aihub_base():
    label_title='模型市场'
    datamodel = SQLAInterface(Aihub)
    base_permissions = ['can_show','can_list']
    base_order = ("id", "desc")
    order_columns = ['id']
    search_columns=['describe','label','name','field','scenes']
    list_columns = ['card']

    spec_label_columns={
        "name":"英文名",
        "field": "领域",
        "label": "中文名",
        "describe":"描述",
        "scenes":"场景",
        "card": "信息"
    }

    edit_form_extra_fields = {
        "field": SelectField(
            label='AI领域',
            description='AI领域',
            widget=Select2Widget(),
            default='',
            choices=[['机器视觉','机器视觉'], ['听觉','听觉'],['自然语言', '自然语言'],['强化学习', '强化学习'],['图论', '图论'], ['通用','通用']]
        ),
    }


class Aihub_visual_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/visual/api'
    base_filters = [["id", Aihub_Filter, 'visual']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_visual_Api)


class Aihub_voice_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/voice/api'
    base_filters = [["id", Aihub_Filter, 'voice']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_voice_Api)


class Aihub_language_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/language/api'
    base_filters = [["id", Aihub_Filter, 'language']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_language_Api)


class Aihub_reinforcement_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/reinforcement/api'
    base_filters = [["id", Aihub_Filter, 'reinforcement']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_reinforcement_Api)

class Aihub_graph_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/graph/api'
    base_filters = [["id", Aihub_Filter, 'graph']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_graph_Api)

class Aihub_common_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/common/api'
    base_filters = [["id", Aihub_Filter, 'common']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_common_Api)



class Aihub_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/api'
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['list_ui_type']='card'
        response['list_ui_args']={
            "card_width":'385px',
            "card_heigh": '250px'
        }


appbuilder.add_api(Aihub_Api)



