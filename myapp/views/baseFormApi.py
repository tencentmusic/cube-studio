import csv
import functools
import json
import logging
import re
import traceback
import urllib.parse
import os
from inspect import isfunction
from sqlalchemy import create_engine
from flask_appbuilder.actions import action
from flask_babel import gettext as __
from flask_appbuilder.actions import ActionItem
from flask import jsonify, request
from flask import flash,g
from flask import current_app, make_response,send_file
from flask.globals import session
from flask_babel import lazy_gettext as _
import jsonschema
from marshmallow import ValidationError
from marshmallow_sqlalchemy.fields import Related, RelatedList
import prison
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.orm.relationships import RelationshipProperty
from werkzeug.exceptions import BadRequest
from marshmallow import validate
from wtforms import validators
from flask_appbuilder._compat import as_unicode
from flask_appbuilder.const import (
    API_ADD_COLUMNS_RES_KEY,
    API_ADD_COLUMNS_RIS_KEY,
    API_ADD_TITLE_RES_KEY,
    API_ADD_TITLE_RIS_KEY,
    API_DESCRIPTION_COLUMNS_RES_KEY,
    API_DESCRIPTION_COLUMNS_RIS_KEY,
    API_EDIT_COLUMNS_RES_KEY,
    API_EDIT_COLUMNS_RIS_KEY,
    API_EDIT_TITLE_RES_KEY,
    API_EDIT_TITLE_RIS_KEY,
    API_FILTERS_RES_KEY,
    API_FILTERS_RIS_KEY,
    API_LABEL_COLUMNS_RES_KEY,
    API_LABEL_COLUMNS_RIS_KEY,
    API_LIST_COLUMNS_RES_KEY,
    API_LIST_COLUMNS_RIS_KEY,
    API_LIST_TITLE_RES_KEY,
    API_LIST_TITLE_RIS_KEY,
    API_ORDER_COLUMN_RIS_KEY,
    API_ORDER_COLUMNS_RES_KEY,
    API_ORDER_COLUMNS_RIS_KEY,
    API_ORDER_DIRECTION_RIS_KEY,
    API_PAGE_INDEX_RIS_KEY,
    API_PAGE_SIZE_RIS_KEY,
    API_PERMISSIONS_RIS_KEY,
    API_SELECT_COLUMNS_RIS_KEY,
    API_SHOW_COLUMNS_RES_KEY,
    API_SHOW_COLUMNS_RIS_KEY,
    API_SHOW_TITLE_RES_KEY,
    API_SHOW_TITLE_RIS_KEY,
    API_URI_RIS_KEY,
)
from flask import (
    abort,
)
from flask_appbuilder.exceptions import FABException, InvalidOrderByColumnFABException
from flask_appbuilder.security.decorators import permission_name, protect,has_access
from flask_appbuilder.api import BaseModelApi,BaseApi,ModelRestApi
from sqlalchemy.sql import sqltypes
from myapp import app, appbuilder,db,event_logger,cache
from myapp.models.favorite import Favorite
conf = app.config

log = logging.getLogger(__name__)
from flask_appbuilder.baseviews import BaseCRUDView, BaseView, expose


def expose(url="/", methods=("GET",)):
    """
        Use this decorator to expose API endpoints on your API classes.

        :param url:
            Relative URL for the endpoint
        :param methods:
            Allowed HTTP methods. By default only GET is allowed.
    """

    def wrap(f):
        if not hasattr(f, "_urls"):
            f._urls = []
        f._urls.append((url, methods))
        return f

    return wrap


def json_response(message,status,result):
    return jsonify(
        {
            "message":message,
            "status":status,
            "result":result
        }
    )


import pysnooper
# @pysnooper.snoop(depth=5)
# 暴露url+视图函数。视图函数会被覆盖，暴露url也会被覆盖
class MyappFormRestApi(BaseView):
    route_base = ''
    order_columns = []
    primary_key = 'id'
    filters = {}
    label_columns = cols_width = description_columns={
    }
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']
    list_title = ''
    add_title = ''
    edit_title = ''
    show_title=''

    ops_link = []
    page_size = 100
    enable_echart = False
    echart_option=None
    columns={}
    show_columns=list_columns=[]
    add_columns = edit_columns = []
    label_title='标题'
    alert_config = {}  # url:function

    def _init_titles(self):
        """
            Init Titles if not defined
        """
        if self.label_title:
            if not self.list_title:
                self.list_title = "遍历 " + self.label_title
            if not self.add_title:
                self.add_title = "添加 " + self.label_title
            if not self.edit_title:
                self.edit_title = "编辑 " + self.label_title
            if not self.show_title:
                self.show_title = "查看 " + self.label_title

        self.title = self.list_title

    # @pysnooper.snoop()
    def _init_properties(self):
        if not self.route_base:
            self.route_base = self.__class__.__name__.lower()
        # 初始化action自耦段
        self.actions = {}
        for attr_name in dir(self):
            func = getattr(self, attr_name)
            if hasattr(func, "_action"):
                action = ActionItem(*func._action, func=func)
                self.actions[action.name] = action

        # 帮助地址
        self.help_url = conf.get('HELP_URL', {}).get(self.__class__.__name__, '')

        # 字典
        for column_name in self.columns:
            if column_name not in self.label_columns:
                self.label_columns[column_name]=self.columns[column_name].label

            if column_name not in self.description_columns:
                self.description_columns[column_name]=self.columns[column_name].description

            if column_name not in self.cols_width:
                self.cols_width[column_name] = {"type": "ellip2", "width": 100}

        # 列表
        if not self.show_columns and self.columns:
            self.show_columns = list(self.columns.keys())
        if not self.list_columns and self.columns:
            self.list_columns = list(self.columns.keys())
        if not self.list_columns and self.label_columns:
            self.list_columns = list(self.label_columns.keys())

        # # 字典列表
        # if not self.add_columns and self.columns:
        #     self.add_columns = list(self.columns.keys())
        #     self.add_columns.remove(self.primary_key)
        # if not self.edit_columns and self.columns:
        #     self.edit_columns = list(self.columns.keys())
        #     self.edit_columns.remove(self.primary_key)


        # 去除不正常的可编辑列
        if self.primary_key in self.edit_columns:
            self.edit_columns.remove(self.primary_key)

        if 'alert_config' not in conf:
            conf['alert_config']={}
        conf['alert_config'].update(self.alert_config)


    # @pysnooper.snoop()
    def __init__(self):
        super(MyappFormRestApi, self).__init__()
        self._init_titles()
        self._init_properties()

    @expose("/_info", methods=["GET"])
    def api_info(self, **kwargs):
        back={
            "cols_width":self.cols_width,
            "label_columns":self.label_columns,
            "list_columns":self.list_columns,
            "label_title":self.label_title,
            "list_title":self.list_title,
            "ops_link":self.ops_link,
            "permissions":self.base_permissions,
            "route_base":"/"+self.route_base.strip('/')+"/",
            "order_columns":self.order_columns,
            "filters":self.filters,
            "page_size":self.page_size,
            "echart":self.enable_echart,
            "action":self.actions,
            "add_columns":[],
            "add_fieldsets":[],
            "add_title":self.add_title,
            "columns_info":{},
            "description_columns":{},
            "edit_columns":[],
            "edit_fieldsets":[],
            "edit_title":self.edit_title,
            "related":{},
            "show_columns":self.show_columns,
            "show_fieldsets":[],
            "show_title":self.show_title,

        }
        return jsonify(back)

    def query_list(self,order_column,order_direction,page_index,page_size,filters=None,**kargs):
        raise NotImplementedError('To be implemented')

    @expose("/", methods=["GET"])
    def api_list(self, **kwargs):
        _response = dict()
        _args = request.json or {}
        _args.update(json.loads(request.args.get('form_data',"{}")))
        _args.update(request.args)

        order_column=_args.get('order_column','')
        order_direction=_args.get('order_direction','asc')
        page_index, page_size = _args.get(API_PAGE_INDEX_RIS_KEY, 0),_args.get(API_PAGE_SIZE_RIS_KEY, self.page_size)


        count, lst = self.query_list(order_column, order_direction, page_index, page_size,filters=None)

        return jsonify({
            "message":"success",
            "status":0,
            "result":{
                "data":lst,
                "count":count
            }
        })


    @expose("/echart", methods=["GET"])
    def echart(self):
        _args = request.json or {}
        _args.update(json.loads(request.args.get('form_data',"{}")))
        _args.update(request.args)

        if self.echart_option:
            try:
                option = self.echart_option()
                if option:
                    return jsonify({
                        "message":"success",
                        "status":0,
                        "result":option
                    })
            except Exception as e:
                print(e)

        return jsonify({})
