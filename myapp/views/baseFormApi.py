import csv
import functools
import json
import logging
from flask_appbuilder.actions import ActionItem
from flask import Markup, Response, current_app, make_response, send_file, flash, g, jsonify, request
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask.globals import session
from flask_appbuilder.const import (

    API_PAGE_INDEX_RIS_KEY,
    API_PAGE_SIZE_RIS_KEY
)
from myapp import app, appbuilder, db, event_logger, cache

conf = app.config


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

import pysnooper
# 在响应体重添加字段和数据
# @pysnooper.snoop()
def merge_response_func(func, key):
    """
        Use this decorator to set a new merging
        response function to HTTP endpoints

        candidate function must have the following signature
        and be childs of BaseApi:
        ```
            def merge_some_function(self, response, rison_args):
        ```

    :param func: Name of the merge function where the key is allowed
    :param key: The key name for rison selection
    :return: None
    """

    def wrap(f):
        if not hasattr(f, "_response_key_func_mappings"):
            f._response_key_func_mappings = dict()
        f._response_key_func_mappings[key] = func
        return f

    return wrap


def json_response(message, status, result):
    return jsonify(
        {
            "message": message,
            "status": status,
            "result": result
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
    label_columns = cols_width = description_columns = {
    }
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']
    list_title = ''
    add_title = ''
    edit_title = ''
    show_title = ''

    ops_link = []
    page_size = 100
    enable_echart = False
    echart_option = None
    columns = {}
    show_columns = list_columns = []
    add_columns = edit_columns = []
    label_title = ''
    alert_config = {}  # url:function
    expand_columns={}

    add_more_info = None

    # 建构响应体
    @staticmethod
    # @pysnooper.snoop()
    def response(code, **kwargs):
        """
            Generic HTTP JSON response method

        :param code: HTTP code (int)
        :param kwargs: Data structure for response (dict)
        :return: HTTP Json response
        """
        # 添flash的信息
        flashes = session.get("_flashes", [])

        # flashes.append((category, message))
        session["_flashes"] = []
        _ret_json = jsonify(kwargs)
        resp = make_response(_ret_json, code)
        flash_json = []
        for f in flashes:
            flash_json.append([f[0], f[1]])
        resp.headers["api_flashes"] = json.dumps(flash_json)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp


    def _init_titles(self):
        """
            Init Titles if not defined
        """
        if self.label_title:
            if not self.list_title:
                self.list_title = __("遍历 ") + self.label_title
            if not self.add_title:
                self.add_title = __("添加 ") + self.label_title
            if not self.edit_title:
                self.edit_title = __("编辑 ") + self.label_title
            if not self.show_title:
                self.show_title = __("查看 ") + self.label_title

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
                # print(self.actions)

        # 帮助地址
        self.help_url = conf.get('HELP_URL', {}).get(self.__class__.__name__.lower().replace('_modelview_api',''), '')
        # 字典
        for column_name in self.columns:
            if column_name not in self.label_columns:
                self.label_columns[column_name] = self.columns[column_name].label

            if column_name not in self.description_columns:
                self.description_columns[column_name] = self.columns[column_name].description

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
            conf['alert_config'] = {}
        conf['alert_config'].update(self.alert_config)

    # @pysnooper.snoop()
    def __init__(self):
        super(MyappFormRestApi, self).__init__()
        self._init_titles()
        self._init_properties()


    @expose("/_info", methods=["GET"])
    def api_info(self, **kwargs):

        # # 前期处理
        # for ops in self.ops_link:
        #     if 'http://' not in ops['url'] and 'https://' not in ops['url']:
        #         host = "http://" + conf['HOST'] if conf['HOST'] else request.host
        #         ops['url'] = host+ops['url']

        actions_info = {}
        for attr_name in self.actions:
            action = self.actions[attr_name]
            actions_info[action.name] = {
                "name": action.name,
                "text": __(action.text),
                "confirmation": __(action.confirmation),
                "icon": action.icon,
                "multiple": action.multiple,
                "single": action.single
            }

        back = {
            "cols_width": self.cols_width,
            "label_columns": self.label_columns,
            "list_columns": self.list_columns,
            "label_title": self.label_title,
            "list_title": self.list_title,
            "ops_link": self.ops_link,
            "permissions": self.base_permissions,
            "route_base": "/" + self.route_base.strip('/') + "/",
            "order_columns": self.order_columns,
            "filters": self.filters,
            "page_size": self.page_size,
            "echart": self.enable_echart,
            "action": actions_info,
            "add_columns": [],
            "add_fieldsets": [],
            "add_title": self.add_title,
            "columns_info": {},
            "description_columns": {},
            "edit_columns": [],
            "edit_fieldsets": [],
            "edit_title": self.edit_title,
            "related": {},
            "show_columns": self.show_columns,
            "show_fieldsets": [],
            "show_title": self.show_title,
            "download_data":False,
            "import_data":False,
            "enable_favorite":False,
            "primary_key":self.primary_key,
            "help_url":self.help_url
        }

        if self.add_more_info:
            try:
                self.add_more_info(back, **kwargs)
            except Exception as e:
                print(e)

        return jsonify(back)

    def query_list(self, order_column, order_direction, page_index, page_size, filters=None, **kargs):
        raise NotImplementedError('To be implemented')

    @expose("/", methods=["GET"])
    def api_list(self, **kwargs):
        _response = dict()
        _args = request.get_json(silent=True) or {}
        _args.update(json.loads(request.args.get('form_data', "{}")))
        _args.update(request.args)

        order_column = _args.get('order_column', '')
        order_direction = _args.get('order_direction', 'asc')
        page_index, page_size = _args.get(API_PAGE_INDEX_RIS_KEY, 0), _args.get(API_PAGE_SIZE_RIS_KEY, self.page_size)

        count, lst = self.query_list(order_column, order_direction, page_index, page_size, filters=None)

        return jsonify({
            "message": "success",
            "status": 0,
            "result": {
                "data": lst,
                "count": count
            }
        })

    @expose("/echart", methods=["GET"])
    # @pysnooper.snoop()
    def echart(self):
        _args = request.get_json(silent=True) or {}
        _args.update(json.loads(request.args.get('form_data', "{}")))
        _args.update(request.args)

        if self.echart_option:
            try:
                option = self.echart_option()
                if option:
                    return jsonify({
                        "message": "success",
                        "status": 0,
                        "result": option
                    })
            except Exception as e:
                print(e)

        return jsonify({})



    @expose("/action/<string:name>/<int:pk>", methods=["GET"])
    def single_action(self, name, pk):
        """
            Action method to handle actions from a show view
        """
        action = self.actions.get(name)
        try:
            res = action.func(pk)
            back = {
                "status": 0,
                "result": {},
                "message": 'success'
            }
            return self.response(200, **back)
        except Exception as e:
            print(e)
            back = {
                "status": -1,
                "message": str(e),
                "result": {}
            }
            return self.response(200, **back)

    @expose("/multi_action/<string:name>", methods=["POST"])
    def multi_action(self, name):
        """
            Action method to handle multiple records selected from a list view
        """
        pks = request.json["ids"]
        action = self.actions.get(name)
        try:
            back = action.func(pks)
            message = back if type(back) == str else 'success'
            back = {
                "status": 0,
                "result": {},
                "message": message
            }
            return self.response(200, **back)
        except Exception as e:
            print(e)
            back = {
                "status": -1,
                "message": str(e),
                "result": {}
            }
            return self.response(200, **back)
