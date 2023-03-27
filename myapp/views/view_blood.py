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
from flask_appbuilder.actions import action
import re,os
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from sqlalchemy.exc import InvalidRequestError
# 将model添加成视图，并控制在前端的显示
from myapp import app, appbuilder,db,event_logger
from myapp.utils import core
from wtforms import BooleanField, IntegerField,StringField, SelectField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp.utils.security_api_impl import grant_hive_access_user
from flask import Flask, jsonify
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
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
from myapp.security import MyUser
conf = app.config
logging = app.logger


class Blood(BaseMyappView):
    route_base='/blood'
    default_view = 'search_node'   # 设置进入蓝图的默认访问视图（没有设置网址的情况下）

    # 探索新节点
    # 搜索新节点
    @expose('/search/node')
    def search_node(self):
        args = request.args

        data={
            "blood":[
                {
                    "node_id": "xx",  # 全网唯一的id，字符串类型
                    "name": 'xx',  # 节点名称
                    "color": "#102030",
                    "shape": "rectangle",
                    "platform": "tdw",     # clickhouse  superset tdw  # 不同平台采集的元数据
                    "node_type": "table",  # 不同平台上不同节点的类型，
                    "lable": ['xx', 'xx', 'xx'],  # 节点显示的标签,没有就不显示
                    "children": {
                        "$node_type": ["$node_id", ]
                    },
                    "parent": {
                        "$node_type": ["$node_id", ]
                    }
                }
            ],
            "control":{
                "node_ops":["detail","explore"],  # 节点可进行的操作  详情查看/节点上下链路探索，以后才有功能再加
                "direction":"horizontal"  # 或者vertical
            }
        }
        return jsonify({
            "status":0,
            "message":"success",
            "result":data
        })



    @expose('/show/<node_id>')
    def show(self,node_id):
        show_info={
            "$tab1":{
                "detail":[
                    {
                        "$group1_name":{
                            "data":{
                              "key1": "value",  # value支持html源码
                              "key2": "value"
                            },
                            "type":"map"  # 字典类型数据
                        },
                        "$group2_name": {
                            "data": 'xxxxx',  # value支持html源码
                            "type": "str"
                        },
                        "$group3_name": {
                            "data": {
                                "url":"http://xxxx/x",
                                "dom":"xx"
                            },
                            "type": "iframe"
                        },
                        "$group4_name": {
                            "data": {
                                "url": "http://xxxx/x",
                                "log": "reserve",  # reserve保留之前的日志，clear 清理之前的日志
                                "cycle": 0,    # 访问的周期秒数，0表示只访问一次
                                "timeout": 300   # 每次访问的超时
                            },
                            "type": "api"
                        }
                    },
                ],
                "link":[
                    {
                        "label":"$xx",
                        "icon":"$<svg ",
                        "url":"$http://xxxxx"
                    },
                ]
            },
            "$tab2":{

            }
        }
        # 返回模板

        return jsonify({
            "status":0,
            "message":"success",
            "result":show_info
        })


# add_view_no_menu添加视图，但是没有菜单栏显示
appbuilder.add_view_no_menu(Blood)




