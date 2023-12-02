import re
import shutil
import time

from flask_appbuilder.models.sqla.interface import SQLAInterface
import urllib.parse
from myapp import app, appbuilder,db
from wtforms import SelectField
from flask_appbuilder.fieldwidgets import Select2Widget
from flask import g,make_response,Markup,jsonify,request
import random,pysnooper,os

from .baseApi import (
    MyappModelRestApi
)
from flask import (
    flash,
    redirect
)
from .base import (
    MyappFilter,
)
from myapp.models.model_aihub import Aihub
from myapp.models.model_notebook import Notebook
from myapp.utils import core
from myapp.utils.py.py_k8s import K8s
from flask_appbuilder import expose
import datetime,json
conf = app.config
logging = app.logger


# 获取某类project分组
class Aihub_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        # user_roles = [role.name.lower() for role in list(get_user_roles())]
        # if "admin" in user_roles:
        #     return query.filter(Project.type == value).order_by(Project.id.desc())
        return query.filter(self.model.field == value)


class Aihub_base():
    label_title = '模型市场'
    datamodel = SQLAInterface(Aihub)
    base_permissions = ['can_show','can_list']
    base_order = ("hot", "desc")
    order_columns = ['id']
    search_columns = ['describe', 'label', 'name', 'scenes']
    list_columns = ['card']
    page_size = 100

    spec_label_columns = {
        "name": "英文名",
        "field": "领域",
        "label": "中文名",
        "describe": "描述",
        "scenes": "场景",
        "card": "信息"
    }

    edit_form_extra_fields = {
        "field": SelectField(
            label='AI领域',
            description='AI领域',
            widget=Select2Widget(),
            default='',
            choices=[['机器视觉', '机器视觉'], ['听觉', '听觉'], ['自然语言', '自然语言'], ['多模态', '多模态'], ['大模型', '大模型']]
        ),
    }


    def post_list(self,items):
        flash('AIHub内容使用，请使用<a target="_blank" href="https://github.com/tencentmusic/cube-studio/blob/master/README_CN.md">企业版</a>',category='success')
        return items


class Aihub_visual_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/visual/api'
    base_filters = [["id", Aihub_Filter, '机器视觉']]

    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }


appbuilder.add_api(Aihub_visual_Api)


class Aihub_voice_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/voice/api'
    base_filters = [["id", Aihub_Filter, '听觉']]

    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }

appbuilder.add_api(Aihub_voice_Api)


class Aihub_language_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/language/api'
    base_filters = [["id", Aihub_Filter, '自然语言']]

    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }

appbuilder.add_api(Aihub_language_Api)


class Aihub_multimodal_Api(Aihub_base,MyappModelRestApi):
    route_base = '/model_market/multimodal/api'
    base_filters = [["id", Aihub_Filter, '多模态']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }

appbuilder.add_api(Aihub_multimodal_Api)


class Aihub_graph_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/graph/api'
    base_filters = [["id", Aihub_Filter, '图论']]

    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }

appbuilder.add_api(Aihub_graph_Api)


class Aihub_common_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/aigc/api'
    base_filters = [["id", Aihub_Filter, '大模型']]

    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        response['list_ui_type'] = 'card'
        response['list_ui_args'] = {
            "card_width": '23%',
            "card_heigh": '250px'
        }

appbuilder.add_api(Aihub_common_Api)


class Aihub_Api(Aihub_base, MyappModelRestApi):
    route_base = '/model_market/all/api'


appbuilder.add_api(Aihub_Api)
