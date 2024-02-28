import copy
import csv
import functools
import json
import logging
import re
import traceback
import urllib.parse
import os
from flask import Markup, Response, current_app, make_response, send_file, flash, g, jsonify, request
from inspect import isfunction

from sqlalchemy import create_engine
from flask_appbuilder.actions import action
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import ActionItem
from flask.globals import session
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
from flask_appbuilder.security.decorators import permission_name, protect, has_access
from flask_appbuilder.api import BaseModelApi, BaseApi, ModelRestApi
from sqlalchemy.sql import sqltypes
from myapp import app, appbuilder, db, event_logger, cache
from myapp.models.favorite import Favorite

conf = app.config

API_COLUMNS_INFO_RIS_KEY = 'columns_info'
API_ADD_FIELDSETS_RIS_KEY = 'add_fieldsets'
API_EDIT_FIELDSETS_RIS_KEY = 'edit_fieldsets'
API_SHOW_FIELDSETS_RIS_KEY = 'show_fieldsets'
API_HELP_URL_RIS_KEY = 'help_url'
API_ACTION_RIS_KEY = 'action'
API_ROUTE_RIS_KEY = 'route_base'

API_PERMISSIONS_RIS_KEY = "permissions"
API_USER_PERMISSIONS_RIS_KEY = "user_permissions"
API_RELATED_RIS_KEY = "related"
API_COLS_WIDTH_RIS_KEY = 'cols_width'
API_EXIST_ADD_ARGS_RIS_KEY = 'exist_add_args'
API_IMPORT_DATA_RIS_KEY = 'import_data'
API_DOWNLOAD_DATA_RIS_KEY = 'download_data'
API_OPS_BUTTON_RIS_KEY = 'ops_link'
API_ENABLE_FAVORITE_RIS_KEY = 'enable_favorite'
API_ECHART = 'echart'


def get_error_msg():
    if current_app.config.get("FAB_API_SHOW_STACKTRACE"):
        return traceback.format_exc()
    return "Fatal error"


def safe(f):
    def wraps(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except BadRequest as e:
            return self.response_error(400, message=str(e))
        except Exception as e:
            logging.exception(e)
            return self.response_error(500, message=get_error_msg())

    return functools.update_wrapper(wraps, f)


def rison(schema=None):
    """
        Use this decorator to parse URI *Rison* arguments to
        a python data structure, your method gets the data
        structure on kwargs['rison']. Response is HTTP 400
        if *Rison* is not correct::

            class ExampleApi(BaseApi):
                    @expose('/risonjson')
                    @rison()
                    def rison_json(self, **kwargs):
                        return self.response(200, result=kwargs['rison'])

        You can additionally pass a JSON schema to
        validate Rison arguments::

            schema = {
                "type": "object",
                "properties": {
                    "arg1": {
                        "type": "integer"
                    }
                }
            }

            class ExampleApi(BaseApi):
                    @expose('/risonjson')
                    @rison(schema)
                    def rison_json(self, **kwargs):
                        return self.response(200, result=kwargs['rison'])
    """

    def _rison(f):
        def wraps(self, *args, **kwargs):
            value = request.args.get(API_URI_RIS_KEY, None)
            kwargs["rison"] = dict()
            if value:
                try:
                    kwargs["rison"] = prison.loads(value)
                except prison.decoder.ParserException:
                    if current_app.config.get("FAB_API_ALLOW_JSON_QS", True):
                        # Rison failed try json encoded content
                        try:
                            kwargs["rison"] = json.loads(
                                urllib.parse.parse_qs(f"{API_URI_RIS_KEY}={value}").get(
                                    API_URI_RIS_KEY
                                )[0]
                            )
                        except Exception:
                            return self.response_error(400, message="Not a valid rison/json argument")
                    else:
                        return self.response_error(400, message="Not a valid rison argument")
            if schema:
                try:
                    jsonschema.validate(instance=kwargs["rison"], schema=schema)
                except jsonschema.ValidationError as e:
                    return self.response_error(400, message=f"Not a valid rison schema {e}")
            return f(self, *args, **kwargs)

        return functools.update_wrapper(wraps, f)

    return _rison


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


# 在响应体重添加字段和数据
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
class MyappModelRestApi(ModelRestApi):
    # 定义主键列
    label_title = ''
    list_title =''
    add_title = ''
    edit_title = ''
    show_title = ''
    primary_key = 'id'
    api_type = 'json'
    allow_browser_login = True
    base_filters = []
    page_size = 100
    src_item_object = None  # 原始model对象
    src_item_json = {}  # 原始model对象的json
    check_edit_permission = None
    check_delete_permission = None
    datamodel = None

    def pre_show(self, item):
        pass

    def pre_show_res(self, result):
        return result

    def pre_add(self, item):
        pass

    def post_add(self, item):
        pass

    def pre_add_req(self, req_json):
        return req_json

    def pre_add_res(self, result):
        return result

    def pre_update_req(self, req_json):
        return req_json

    def pre_update_res(self, result):
        return result

    def pre_update(self, item):
        pass

    def post_update(self, item):
        pass

    def pre_list_req(self, req_json):
        return req_json

    def pre_list_res(self, result):
        return result

    # 可以调整顺序
    def post_list(self, items):
        return items

    def pre_delete(self, item):
        pass

    def post_delete(self, item):
        pass

    # 添加和更新前info信息的查询，或者填写界面的查询
    def pre_add_web(self):
        pass

    def pre_update_web(self, item):
        pass

    edit_form_extra_fields = {}
    add_form_extra_fields = {}
    add_fieldsets = []
    edit_fieldsets = []
    show_fieldsets = []

    ops_link = [
        # {
        #     "text": "git",
        #     "url": "https://github.com/tencentmusic/cube-studio"
        # }
    ]

    help_url = None

    default_filter = {}
    actions = {}

    user_permissions = {
        "add": True,
        "edit": True,
        "delete": True,
        "show": True
    }
    add_form_query_rel_fields = {}
    edit_form_query_rel_fields = {}
    related_views = []
    add_more_info = None
    remember_columns = []
    spec_label_columns = {}
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']
    cols_width = {}
    import_data = False
    download_data = False
    enable_favorite = False
    enable_echart = False
    pre_upload = None
    set_columns_related = None
    echart_option = None
    alert_config = {}
    expand_columns = {}
    order_columns = []

    # @pysnooper.snoop()
    def csv_response(self, file_path, file_name=None):
        # 下载csv
        response = make_response(send_file(file_path, as_attachment=True, conditional=True))
        if not file_name:
            file_name = os.path.basename(file_path)
        if '.csv' not in file_name:
            file_name = file_name + ".csv"
        response.headers["Content-Disposition"] = f"attachment; filename={file_name}".format(file_name=file_name)
        return response

    # 流式读取
    def send_file(self, store_path):
        with open(store_path, 'rb') as targetfile:
            while 1:
                data = targetfile.read(20 * 1024 * 1024)  # 每次读取20M
                if not data:
                    break
                yield data

    # 建构响应体
    @staticmethod
    # @pysnooper.snoop(watch_explode='kwargs')
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
        _ret_json='unknow error'
        if 'message' in kwargs and 'status' in kwargs and kwargs['status']!=0 and type(kwargs['message'])==str:
            _ret_json = json.dumps(kwargs,ensure_ascii=False,indent=4)
        else:
            _ret_json = jsonify(kwargs)
        resp = make_response(_ret_json, code)
        flash_json = []
        for f in flashes:
            flash_json.append([f[0], f[1]])
        resp.headers["api_flashes"] = json.dumps(flash_json)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp

    # @pysnooper.snoop()
    def _init_titles(self):

        if self.label_title:
            # if not self.list_title:
            self.list_title = self.label_title + __(" 列表")
            # if not self.add_title:
            self.add_title = __("添加 ") + self.label_title
            # if not self.edit_title:
            self.edit_title = __("编辑 ") + self.label_title
            # if not self.show_title:
            self.show_title = __("查看 ") + self.label_title

        self.title = self.list_title

    # @pysnooper.snoop()
    def _init_label_columns(self):
        # 初始化label字段
        # 全局的label
        all_label_columns={}
        if hasattr(self.datamodel.obj, 'label_columns') and self.datamodel.obj.label_columns:
            for col in self.datamodel.obj.label_columns:
                all_label_columns[col] = self.datamodel.obj.label_columns[col]
        # print(all_label_columns)
        new_label_columns = self.list_columns+self.show_columns+self.edit_columns+self.add_columns
        new_label_columns = list(set(new_label_columns))
        for col in new_label_columns:
            self.label_columns[col] = all_label_columns.get(col,col)
            # print(col,self.label_columns[col])

        # 本view特定的label
        for col in self.spec_label_columns:
            self.label_columns[col] = self.spec_label_columns[col]


    # @pysnooper.snoop()
    def _init_properties(self):
        """
            Init Properties
        """
        super(MyappModelRestApi, self)._init_properties()

        # 初始化action自耦段
        self.actions = {}
        for attr_name in dir(self):
            func = getattr(self, attr_name)
            if hasattr(func, "_action"):
                action = ActionItem(*func._action, func=func)
                self.actions[action.name] = action

        # self._init_label_columns()
        self.primary_key = self.datamodel.get_pk_name()

        self._init_cols_width()

        # 帮助地址
        self.help_url = conf.get('HELP_URL', {}).get(self.datamodel.obj.__tablename__, '') if self.datamodel else ''

        # 配置搜索转换器
        self._filters = self.datamodel.get_filters(self.search_columns)

        if 'alert_config' not in conf:
            conf['alert_config'] = {}
        conf['alert_config'].update(self.alert_config)

    def _init_model_schemas(self):
        # Create Marshmalow schemas if one is not specified

        # for column_name in self.edit_columns:
        #     if column_name not in self.add_columns:
        #         self.add_columns.append(column_name)

        if self.list_model_schema is None:
            self.list_model_schema = self.model2schemaconverter.convert(
                self.list_columns
            )
        if self.add_model_schema is None:
            self.add_model_schema = self.model2schemaconverter.convert(
                self.add_columns, nested=False
            )
        if self.edit_model_schema is None:
            self.edit_model_schema = self.model2schemaconverter.convert(
                list(set(list(self.edit_columns+self.show_columns+self.list_columns+self.search_columns))), nested=False
            )
        if self.show_model_schema is None:
            self.show_model_schema = self.model2schemaconverter.convert(
                self.show_columns
            )

    # @pysnooper.snoop(watch_explode=('value','column_type'))
    def _init_cols_width(self):
        # return
        columns = self.datamodel.obj.__table__._columns
        for column in columns:
            if column.name not in self.cols_width and column.name in self.list_columns:  # 只需要配置没有配置过的list_columns
                column_type = column.type
                if column_type.__class__.__name__ in ['Integer', 'Float', 'Numeric', 'Integer', 'Date', 'Enum']:
                    self.cols_width[column.name] = {
                        "type": 'ellip2',
                        "width": 100
                    }
                elif column_type.__class__.__name__ in ['Time', 'Datetime']:
                    self.cols_width[column.name] = {
                        "type": 'ellip2',
                        "width": 300
                    }
                elif column_type.__class__.__name__ in ['String', 'Text']:
                    width = 100
                    if column_type.length and column_type.length > 100 and column_type.length < 500:
                        width = column_type.length
                    if column_type.length and column_type.length > 500:
                        width = 500

                    self.cols_width[column.name] = {
                        "type": 'ellip2',
                        "width": width
                    }

        for attr in self.list_columns:
            if attr not in self.cols_width:
                self.cols_width[attr] = {
                    "type": 'ellip2',
                    "width": 100
                }

        # 固定常用的几个字段的宽度
        # print(self.cols_width)

    # 将列宽信息加入
    def merge_cols_width(self, response, **kwargs):
        response[API_COLS_WIDTH_RIS_KEY] = self.cols_width

    # 将是否批量导入加入
    def merge_ops_data(self, response, **kwargs):
        response[API_IMPORT_DATA_RIS_KEY] = self.import_data
        response[API_DOWNLOAD_DATA_RIS_KEY] = self.download_data
        # for ops in self.ops_link:
        #     if 'http://' not in ops['url'] and 'https://' not in ops['url']:
        #         host = "http://" + conf['HOST'] if conf['HOST'] else request.host
        #         ops['url'] = host+ops['url']

        response[API_OPS_BUTTON_RIS_KEY] = self.ops_link
        response[API_ENABLE_FAVORITE_RIS_KEY] = self.enable_favorite
        response[API_ECHART] = self.enable_echart
        response['page_size'] = self.page_size

    # 重新渲染add界面
    # @pysnooper.snoop()
    def merge_exist_add_args(self, response, **kwargs):
        exist_add_args = request.args.get('exist_add_args', '')
        if exist_add_args:
            exist_add_args = json.loads(exist_add_args)
            # 把这些值转为add_column中的默认值
            # print(response[API_ADD_COLUMNS_RIS_KEY])
            response_add_columns = {}
            for column in response[API_ADD_COLUMNS_RIS_KEY]:
                if column['name'] in exist_add_args and exist_add_args[column['name']]:
                    column['default'] = exist_add_args[column['name']]
                response_add_columns[column['name']] = column
            # 提供字段变换内容
            if self.set_columns_related:
                try:
                    self.set_columns_related(exist_add_args, response_add_columns)
                    response[API_ADD_COLUMNS_RIS_KEY] = list(response_add_columns.values())
                except Exception as e:
                    print(e)

    # 根据columnsfields 转化为 info的json信息
    # @pysnooper.snoop()
    def columnsfield2info(self, columnsfields):
        ret = list()
        for col_name in columnsfields:
            column_field = columnsfields[col_name]
            # print(column_field)
            col_info = {}
            col_info['name'] = col_name

            column_field_kwargs = column_field.kwargs
            # type 类型 EnumField   values
            # aa = column_field
            col_info['type'] = column_field.field_class.__name__.replace('Field', '')
            # ret['description']=column_field_kwargs.get('description','')
            col_info['description'] = self.description_columns.get(col_name, column_field_kwargs.get('description', ''))
            col_info['label'] = self.label_columns.get(col_name, column_field_kwargs.get('label', ''))
            col_info['default'] = column_field_kwargs.get('default', '')
            col_info['validators'] = column_field_kwargs.get('validators', [])
            col_info['choices'] = column_field_kwargs.get('choices', [])
            if 'widget' in column_field_kwargs:
                col_info['widget'] = column_field_kwargs['widget'].__class__.__name__.replace('Widget', '').replace('Field', '').replace('My', '')
                if hasattr(column_field_kwargs['widget'], 'readonly') and column_field_kwargs['widget'].readonly:
                    col_info['disable'] = True
                # 处理可选可填类型
                if hasattr(column_field_kwargs['widget'], 'can_input') and column_field_kwargs['widget'].can_input:
                    col_info['ui-type'] = 'input-select'
                # 处理时间类型
                if hasattr(column_field_kwargs['widget'], 'is_date') and column_field_kwargs['widget'].is_date:
                    col_info['ui-type'] = 'datePicker'
                # 处理时间类型
                if hasattr(column_field_kwargs['widget'], 'is_date_range') and column_field_kwargs['widget'].is_date_range:
                    col_info['ui-type'] = 'rangePicker'

            col_info = self.make_ui_info(col_info)
            ret.append(col_info)
        return ret

    # 每个用户对当前记录的权限，base_permissions 是对所有记录的权限
    def check_item_permissions(self, item):
        self.user_permissions = {
            "add": True,
            "edit": True,
            "delete": True,
            "show": True
        }

    def merge_base_permissions(self, response, **kwargs):
        response[API_PERMISSIONS_RIS_KEY] = [
            permission
            for permission in self.base_permissions
            # if self.appbuilder.sm.has_access(permission, self.class_permission_name)
        ]

    # @pysnooper.snoop()
    def merge_user_permissions(self, response, **kwargs):
        # print(self.user_permissions)
        response[API_USER_PERMISSIONS_RIS_KEY] = self.user_permissions

    # @pysnooper.snoop()
    def merge_expand_field_info(self,columns):
        # print(len(columns))
        if not self.expand_columns:
            return columns
        new_columns=[]
        # 将原本扩展字段替换为扩展后字段
        for index,column in enumerate(columns):
            if column['name'] not in self.expand_columns:
                new_columns.append(column)
            else:
                expand_add_columns = self.expand_columns[column['name']]
                for column_name in expand_add_columns:
                    # print(column_name)
                    expand_add_column = expand_add_columns[column_name]
                    # print(expand_add_column)
                    ret = self.filed2ui(column_name,expand_add_column)
                    # print(ret)
                    ret1 = self.make_ui_info(ret)
                    ret1['name']=column_name
                    # print(ret1)
                    new_columns.append(copy.deepcopy(ret1))   # 要深度复制，不然都变为了最后一个字段
        # print(len(new_columns))
        return new_columns

    # add_form_extra_fields  里面的字段要能拿到才对
    # @pysnooper.snoop(watch_explode=())
    def merge_add_field_info(self, response, **kwargs):
        _kwargs = kwargs.get("add_columns", {})
        # 将关联字段的查询限制条件加入
        if self.add_form_query_rel_fields:
            self.add_query_rel_fields = self.add_form_query_rel_fields

        add_columns = self._get_fields_info(
            self.add_columns,
            self.add_model_schema,
            self.add_query_rel_fields,
            **_kwargs,
        )
        add_columns = self.merge_expand_field_info(add_columns)

        response[API_ADD_COLUMNS_RES_KEY] = add_columns

    # @pysnooper.snoop(watch_explode=('edit_columns'))
    def merge_edit_field_info(self, response, **kwargs):
        _kwargs = kwargs.get("edit_columns", {})
        if self.edit_form_query_rel_fields:
            self.edit_query_rel_fields = self.edit_form_query_rel_fields
        edit_columns = self._get_fields_info(
            self.edit_columns,
            self.edit_model_schema,
            self.edit_query_rel_fields,
            **_kwargs,
        )
        # 处理retry_info，如果有这种类型，就禁止编辑
        for column in edit_columns:
            if column.get('retry_info', False):
                column['disable'] = True
        edit_columns = self.merge_expand_field_info(edit_columns)
        response[API_EDIT_COLUMNS_RES_KEY] = edit_columns

    # @pysnooper.snoop(watch_explode=('edit_columns'))
    def merge_add_fieldsets_info(self, response, **kwargs):
        # if self.pre_add_web:
        #     self.pre_add_web()
        add_fieldsets = []
        if self.add_fieldsets:
            for group in self.add_fieldsets:
                group_name = group[0]
                group_fieldsets = group[1]
                add_fieldsets.append({
                    "group": group_name,
                    "expanded": group_fieldsets.get('expanded', True),
                    "fields": group_fieldsets['fields']
                })

        response[API_ADD_FIELDSETS_RIS_KEY] = add_fieldsets

    # @pysnooper.snoop(watch_explode=('edit_columns'))
    def merge_edit_fieldsets_info(self, response, **kwargs):
        edit_fieldsets = []
        if self.edit_fieldsets:
            for group in self.edit_fieldsets:
                group_name = group[0]
                group_fieldsets = group[1]
                edit_fieldsets.append({
                    "group": group_name,
                    "expanded": group_fieldsets.get('expanded', True),
                    "fields": group_fieldsets['fields']
                })
        response[API_EDIT_FIELDSETS_RIS_KEY] = edit_fieldsets

    def merge_show_fieldsets_info(self, response, **kwargs):
        show_fieldsets = []
        if self.show_fieldsets:
            for group in self.show_fieldsets:
                group_name = group[0]
                group_fieldsets = group[1]
                show_fieldsets.append({
                    "group": group_name,
                    "expanded": group_fieldsets.get('expanded', True),
                    "fields": group_fieldsets['fields']
                })
        response[API_SHOW_FIELDSETS_RIS_KEY] = show_fieldsets

    # @pysnooper.snoop()
    def merge_search_filters(self, response, **kwargs):
        # Get possible search fields and all possible operations
        search_filters = dict()
        dict_filters = self._filters.get_search_filters()
        for col in self.search_columns:
            search_filters[col] = {}
            search_filters[col]['filter'] = [
                {"name": as_unicode(flt.name), "operator": flt.arg_name}
                for flt in dict_filters[col]
            ]

            # print(col)
            # print(self.datamodel.list_columns)
            # 对于外键全部可选值返回，或者还需要现场查询(现场查询用哪个字段是个问题)
            if self.datamodel and self.edit_model_schema:  # 根据edit_column 生成的model_schema，编辑里面才会读取外键对象列表
                # ao = self.edit_model_schema.fields
                if col in self.edit_model_schema.fields:

                    field = self.edit_model_schema.fields[col]
                    # print(field)
                    if isinstance(field, Related) or isinstance(field, RelatedList):
                        filter_rel_field = self.edit_query_rel_fields.get(col, [])
                        # 获取外键对象list
                        search_filters[col]["count"], search_filters[col]["values"] = self._get_list_related_field(
                            field, filter_rel_field, page=0, page_size=10000
                        )
                        # if col in self.datamodel.list_columns:
                        #     search_filters[col]["type"] = self.datamodel.list_columns[col].type

                    search_filters[col]["type"] = field.__class__.__name__ if 'type' not in search_filters[col] else search_filters[col]["type"]

            # 用户可能会自定义字段的操作格式，比如字符串类型，显示和筛选可能是menu
            if col in self.edit_form_extra_fields:
                column_field = self.edit_form_extra_fields[col]
                column_field_kwargs = column_field.kwargs
                # type 类型 EnumField   values
                # aa = column_field
                search_filters[col]['type'] = column_field.field_class.__name__.replace('Field', '').replace('My', '')
                search_filters[col]['choices'] = column_field_kwargs.get('choices', [])
                # 选-填 字段在搜索时为填写字段
                if hasattr(column_field_kwargs.get('widget', {}), 'can_input') and column_field_kwargs['widget'].can_input:
                    search_filters[col]['ui-type'] = 'input'
                # 对于那种配置使用过往记录作为可选值的参数进行处理
                if hasattr(column_field_kwargs.get('widget', {}), 'conten2choices') and column_field_kwargs['widget'].conten2choices:
                    # 先从缓存中拿结果
                    field_contents = None
                    try:
                        field_contents = cache.get(self.datamodel.obj.__tablename__ + "_" + col)
                    except Exception as e:
                        print(e)
                    # 缓存没有数据，再从数据库中读取
                    if not field_contents:
                        try:
                            field_contents = db.session.query(getattr(self.datamodel.obj, col)).group_by(getattr(self.datamodel.obj, col)).all()
                            field_contents = list(set([item[0] for item in field_contents]))
                            cache.set(self.datamodel.obj.__tablename__ + "_" + col, field_contents, timeout=60 * 60)

                        except Exception as e:
                            print(e)

                    if field_contents:
                        search_filters[col]['ui-type'] = 'select'
                        search_filters[col]['choices'] = [[x, x] for x in list(set(field_contents))]

            search_filters[col] = self.make_ui_info(search_filters[col])
            # 多选字段在搜索时为单选字段
            if search_filters[col].get('ui-type', '') == 'select2':
                search_filters[col]['ui-type'] = 'select'

            search_filters[col]['default'] = self.default_filter.get(col, '')
        response[API_FILTERS_RES_KEY] = search_filters

    def merge_add_title(self, response, **kwargs):
        response[API_ADD_TITLE_RES_KEY] = self.add_title

    def merge_edit_title(self, response, **kwargs):
        response[API_EDIT_TITLE_RES_KEY] = self.edit_title

    def merge_label_columns(self, response, **kwargs):
        _pruned_select_cols = kwargs.get(API_SELECT_COLUMNS_RIS_KEY, [])
        if _pruned_select_cols:
            columns = _pruned_select_cols
        else:
            # Send the exact labels for the caller operation
            if kwargs.get("caller") == "list":
                columns = self.list_columns
            elif kwargs.get("caller") == "show":
                columns = self.show_columns
            else:
                columns = self.label_columns  # pragma: no cover
        response[API_LABEL_COLUMNS_RES_KEY] = self._label_columns_json(columns)

    def merge_list_label_columns(self, response, **kwargs):
        self.merge_label_columns(response, caller="list", **kwargs)

    def merge_show_label_columns(self, response, **kwargs):
        self.merge_label_columns(response, caller="show", **kwargs)

    # @pysnooper.snoop()
    def merge_show_columns(self, response, **kwargs):
        _pruned_select_cols = kwargs.get(API_SELECT_COLUMNS_RIS_KEY, [])
        if _pruned_select_cols:
            response[API_SHOW_COLUMNS_RES_KEY] = _pruned_select_cols
        else:
            response[API_SHOW_COLUMNS_RES_KEY] = self.show_columns

    def merge_description_columns(self, response, **kwargs):
        _pruned_select_cols = kwargs.get(API_SELECT_COLUMNS_RIS_KEY, [])
        if _pruned_select_cols:
            response[API_DESCRIPTION_COLUMNS_RES_KEY] = self._description_columns_json(
                _pruned_select_cols
            )
        else:
            # Send all descriptions if cols are or request pruned
            response[API_DESCRIPTION_COLUMNS_RES_KEY] = self._description_columns_json(
                self.description_columns
            )

    def merge_list_columns(self, response, **kwargs):
        _pruned_select_cols = kwargs.get(API_SELECT_COLUMNS_RIS_KEY, [])
        if _pruned_select_cols:
            response[API_LIST_COLUMNS_RES_KEY] = _pruned_select_cols
        else:
            response[API_LIST_COLUMNS_RES_KEY] = self.list_columns

    def merge_order_columns(self, response, **kwargs):
        _pruned_select_cols = kwargs.get(API_SELECT_COLUMNS_RIS_KEY, [])
        if _pruned_select_cols:
            response[API_ORDER_COLUMNS_RES_KEY] = [
                order_col
                for order_col in self.order_columns
                if order_col in _pruned_select_cols
            ]
        else:
            response[API_ORDER_COLUMNS_RES_KEY] = self.order_columns

    # @pysnooper.snoop(watch_explode=('aa'))
    def merge_columns_info(self, response, **kwargs):
        columns_info = {}
        for attr in dir(self.datamodel.obj):
            value = getattr(self.datamodel.obj, attr) if hasattr(self.datamodel.obj, attr) else None
            if type(value) == InstrumentedAttribute:
                if type(value.comparator) == ColumnProperty.Comparator:
                    columns_info[value.key] = {
                        "type": str(value.comparator.type)
                    }
                if type(value.comparator) == RelationshipProperty.Comparator:
                    columns_info[value.key] = {
                        "type": "Relationship"
                    }
                    class_name = value.comparator.property.argument.__class__.__name__
                    if 'ModelDeclarativeMeta' in class_name:
                        columns_info[value.key]['relationship'] = value.comparator.property.argument.__name__
                    elif '_class_resolver' in class_name:
                        columns_info[value.key]['relationship'] = value.comparator.property.argument.arg

        response[API_COLUMNS_INFO_RIS_KEY] = columns_info

    def merge_help_url_info(self, response, **kwargs):
        response[API_HELP_URL_RIS_KEY] = self.help_url

    # @pysnooper.snoop(watch_explode='aa')
    def merge_action_info(self, response, **kwargs):
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
        response[API_ACTION_RIS_KEY] = actions_info

    def merge_route_info(self, response, **kwargs):
        response[API_ROUTE_RIS_KEY] = "/" + self.route_base.strip('/') + "/"
        response['primary_key'] = self.primary_key
        response['label_title'] = self.label_title or self._prettify_name(self.datamodel.model_name)

    # @pysnooper.snoop(watch_explode=())
    # 添加关联model的字段
    def merge_related_field_info(self, response, **kwargs):
        try:
            add_info = {}
            if self.related_views:
                for related_views_class in self.related_views:
                    related_views = related_views_class()
                    related_views._init_model_schemas()
                    if related_views.add_form_query_rel_fields:
                        related_views.add_query_rel_fields = related_views.add_form_query_rel_fields

                    # print(related_views.add_columns)
                    # print(related_views.add_model_schema)
                    # print(related_views.add_query_rel_fields)
                    add_columns = related_views._get_fields_info(
                        cols=related_views.add_columns,
                        model_schema=related_views.add_model_schema,
                        filter_rel_fields=related_views.add_query_rel_fields,
                        **kwargs,
                    )
                    add_info[str(related_views.datamodel.obj.__name__).lower()] = add_columns
                    # add_info[related_views.__class__.__name__]=add_columns
            response[API_RELATED_RIS_KEY] = add_info
        except Exception as e:
            print(e)
        pass

    def merge_list_title(self, response, **kwargs):
        self._init_titles()
        # print(self.list_title)
        response[API_LIST_TITLE_RES_KEY] = self.list_title

    def merge_show_title(self, response, **kwargs):
        response[API_SHOW_TITLE_RES_KEY] = self.show_title

    def merge_more_info(self, response, **kwargs):
        # 将 配置根据历史填写值作为选项的字段配置出来。
        if self.add_more_info:
            try:
                self.add_more_info(response, **kwargs)
            except Exception as e:
                print(e)

    def response_error(self, code, message='error', status=1, result={}):
        back_data = {
            'result': result,
            "status": status,
            'message': message
        }
        return self.response(code, **back_data)

    @expose("/_info", methods=["GET"])
    @merge_response_func(merge_more_info, 'more_info')
    @merge_response_func(merge_ops_data, API_IMPORT_DATA_RIS_KEY)
    @merge_response_func(merge_exist_add_args, API_EXIST_ADD_ARGS_RIS_KEY)
    @merge_response_func(merge_cols_width, API_COLS_WIDTH_RIS_KEY)
    @merge_response_func(merge_base_permissions, API_PERMISSIONS_RIS_KEY)
    @merge_response_func(merge_user_permissions, API_USER_PERMISSIONS_RIS_KEY)
    @merge_response_func(merge_add_field_info, API_ADD_COLUMNS_RIS_KEY)
    @merge_response_func(merge_edit_field_info, API_EDIT_COLUMNS_RIS_KEY)
    @merge_response_func(merge_add_fieldsets_info, API_ADD_FIELDSETS_RIS_KEY)
    @merge_response_func(merge_edit_fieldsets_info, API_EDIT_FIELDSETS_RIS_KEY)
    @merge_response_func(merge_show_fieldsets_info, API_SHOW_FIELDSETS_RIS_KEY)
    @merge_response_func(merge_search_filters, API_FILTERS_RIS_KEY)
    @merge_response_func(merge_show_label_columns, API_LABEL_COLUMNS_RIS_KEY)
    @merge_response_func(merge_show_columns, API_SHOW_COLUMNS_RIS_KEY)
    @merge_response_func(merge_list_label_columns, API_LABEL_COLUMNS_RIS_KEY)
    @merge_response_func(merge_list_columns, API_LIST_COLUMNS_RIS_KEY)
    @merge_response_func(merge_show_title, API_SHOW_TITLE_RIS_KEY)
    @merge_response_func(merge_add_title, API_ADD_TITLE_RIS_KEY)
    @merge_response_func(merge_edit_title, API_EDIT_TITLE_RIS_KEY)
    @merge_response_func(merge_list_title, API_LIST_TITLE_RIS_KEY)
    @merge_response_func(merge_description_columns, API_DESCRIPTION_COLUMNS_RIS_KEY)
    @merge_response_func(merge_order_columns, API_ORDER_COLUMNS_RIS_KEY)
    @merge_response_func(merge_columns_info, API_COLUMNS_INFO_RIS_KEY)
    @merge_response_func(merge_help_url_info, API_HELP_URL_RIS_KEY)
    @merge_response_func(merge_action_info, API_ACTION_RIS_KEY)
    @merge_response_func(merge_route_info, API_ROUTE_RIS_KEY)
    @merge_response_func(merge_related_field_info, API_RELATED_RIS_KEY)
    def api_info(self, **kwargs):
        _response = dict()
        _args = kwargs.get("rison", {})
        _args.update(request.args)
        id = _args.get(self.primary_key, '')
        if id:
            item = self.datamodel.get(id)
            if item and self.pre_update_web:
                try:
                    self.pre_update_web(item)
                except Exception as e:
                    print(e)
            # # 似乎未生效
            # if item and self.check_item_permissions:
            #     try:
            #         self.check_item_permissions(item)
            #     except Exception as e:
            #         print(e)

            if self.check_edit_permission:
                has_permission = self.check_edit_permission(item)
                if not has_permission:
                    flash('no permission to edit', category='warning')
                    return self.response_error(422, message='no permission to edit')

        elif self.pre_add_web:
            try:
                self.pre_add_web()
            except Exception as e:
                print(e)

        self.set_response_key_mappings(_response, self.api_info, _args, **_args)
        return self.response(200, **_response)

    # 将expand类型的字段展开
    # @pysnooper.snoop(watch_explode=('json_data'))
    def show_expand(self,json_data_src):
        json_data = copy.deepcopy(json_data_src)
        for column_name in self.expand_columns:
            new_columns = list(self.expand_columns[column_name].keys())
            if json_data[column_name]:
                expand_value = json.loads(json_data[column_name])  # 在原来基础上，修改扩展出来的几个字段
            else:
                expand_value = {}
            for col in new_columns:
                json_data[col] = expand_value.get(col,None)

        return json_data

    @expose("/<int:pk>", methods=["GET"])
    # @pysnooper.snoop(depth=4)
    def api_show(self, pk, **kwargs):

        # from flask_appbuilder.models.sqla.interface import SQLAInterface
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            return self.response_error(404, "Not found")

        if self.pre_show:
            self.pre_show(item)

        _response = dict()
        _args = kwargs.get("rison", {})
        if 'form_data' in request.args:
            _args.update(json.loads(request.args.get('form_data')))

        for key in request.args:
            if key!='form_data':
                _args[key.replace('form_data','').replace('[','').replace(']','')]=request.args[key]


        select_cols = _args.get(API_SELECT_COLUMNS_RIS_KEY, [])
        _pruned_select_cols = [col for col in select_cols if col in self.show_columns]
        self.set_response_key_mappings(
            _response,
            self.get,
            _args,
            **{API_SELECT_COLUMNS_RIS_KEY: _pruned_select_cols},
        )
        if _pruned_select_cols:
            _show_model_schema = self.model2schemaconverter.convert(_pruned_select_cols)
        else:
            _show_model_schema = self.show_model_schema

        data = _show_model_schema.dump(item, many=False)
        if int(_args.get('str_related', 0)):
            for key in data:
                if type(data[key]) == dict:
                    data[key] = str(getattr(item, key))

        # 按show_columns的顺序显示
        data = sorted(data.items(),key=lambda kv: self.show_columns.index(kv[0]) if kv[0] in self.show_columns else 1000)
        data = dict(zip([x[0] for x in data], [x[1] for x in data]))

        _response['data'] = data  # item.to_json()
        _response['data'][self.primary_key] = pk
        if self.expand_columns:
            _response['data'] = self.show_expand(_response['data'])  # 将扩展字段展开显示
        back = self.pre_show_res(_response)
        back_data = {
            'result': back['data'] if back else _response['data'],
            "status": 0,
            'message': "success"
        }

        return self.response(200, **back_data)

    @expose("/", methods=["GET"])
    # @pysnooper.snoop(watch_explode=('aa'))
    def api_list(self, **kwargs):
        _response = dict()

        try:
            req_json = request.get_json(silent=True) or {}
        except Exception as e:
            print(e)
            req_json = {}

        _args = req_json or {}
        _args.update(request.args)
        # 应对那些get无法传递body的请求，也可以把body放在url里面
        if 'form_data' in request.args:
            _args.update(json.loads(request.args.get('form_data')))

        for key in request.args:
            if key!='form_data':
                _args[key.replace('form_data','').replace('[','').replace(']','')]=request.args[key]

        if self.pre_list_req:
            _args = self.pre_list_req(_args)

        # handle select columns
        select_cols = _args.get(API_SELECT_COLUMNS_RIS_KEY, [])
        _pruned_select_cols = [col for col in select_cols if col in list(set(self.list_columns + self.show_columns))]
        self.set_response_key_mappings(
            _response,
            self.get_list,
            _args,
            **{API_SELECT_COLUMNS_RIS_KEY: _pruned_select_cols},
        )

        if _pruned_select_cols:
            _list_model_schema = self.model2schemaconverter.convert(_pruned_select_cols)
        else:
            _list_model_schema = self.list_model_schema
        # handle filters
        try:
            # 参数缩写都在每个filter的arg_name
            joined_filters = self._handle_filters_args(_args)
        except FABException as e:
            return self.response_error(400, message=str(e))
        # handle base order
        try:
            order_column, order_direction = self._handle_order_args(_args)
        except InvalidOrderByColumnFABException as e:
            return self.response_error(400, message=str(e))
        # handle pagination
        page_index, page_size = self._handle_page_args(_args)
        # Make the query
        query_select_columns = _pruned_select_cols or self.list_columns

        # @pysnooper.snoop()
        def get_favorite():
            all_rows = db.session.query(Favorite.row_id).filter(Favorite.model_name == self.datamodel.obj.__tablename__).filter(Favorite.user_id == g.user.id).all()
            all_rows_id = [int(row[0]) for row in all_rows]
            return all_rows_id

        # 如果只查询收藏
        if _args.get('only_favorite', False):
            if self.primary_key not in self.search_columns:
                self.search_columns.append(self.primary_key)
            _args['filters'] = []
            self._filters.clear_filters()
            from flask_appbuilder.models.sqla.filters import FilterInFunction
            self._filters.add_filter(self.primary_key, FilterInFunction, get_favorite)
            joined_filters = self._filters.get_joined_filters(self._base_filters)  # 将基础filter加入到过滤器中你那个

        count, lst = self.datamodel.query(
            joined_filters,
            order_column,
            order_direction,
            page=page_index,
            page_size=page_size,
            select_columns=query_select_columns,
        )
        if self.post_list:
            lst = self.post_list(lst)

        # pks = self.datamodel.get_keys(lst)
        # import marshmallow.schema
        # for item in lst:
        #     if self.datamodel.is_relation(item)
        # aa =
        # item.project = 'aaa'

        data = _list_model_schema.dump(lst, many=True)

        # 把外键换成字符串
        if int(_args.get('str_related', 0)):
            for index in range(len(data)):
                for key in data[index]:
                    if type(data[index][key]) == dict:
                        data[index][key] = str(getattr(lst[index], key))

        _response['data'] = data  # [item.to_json() for item in lst]
        # _response["ids"] = pks
        _response["count"] = count  # 这个是总个数
        for index in range(len(lst)):
            _response['data'][index][self.primary_key] = getattr(lst[index], self.primary_key)
        result = _response
        try:
            result = self.pre_list_res(_response)
        except Exception as e:
            print(e)

        back_data = {
            'result': result,
            "status": 0,
            'message': "success"
        }
        # print(back_data)
        # aa = back_data['result']
        # back_data['result']['data'][0]['creator']='aaaa'
        return self.response(200, **back_data)

    # @pysnooper.snoop()
    def json_to_item(self, data):
        class Back:
            pass

        back = Back()
        try:
            item = self.datamodel.obj(**data)
            # for key in data:
            #     if hasattr(item,key):
            #         setattr(item,key,data[key])

            setattr(back, 'data', item)
        except Exception as e:
            setattr(back, 'data', data)
            setattr(back, 'errors', str(e))
        return back

    # 将扩展出来的字段合并为源字段，更新源地址，并不替换原地址
    # @pysnooper.snoop(watch_explode='json_data')
    def to_expand(self,json_data_src):
        json_data = copy.deepcopy(json_data_src)
        for column_name in self.expand_columns:
            new_columns = list(self.expand_columns[column_name].keys())
            if self.src_item_json.get(column_name,None):
                expand_value=json.loads(self.src_item_json[column_name])  # 在原来基础上，修改扩展出来的几个字段
            else:
                expand_value = {}
            for col in new_columns:
                if col in json_data and json_data[col]!=None:  # 必须有值才记录
                    expand_value[col] = json_data[col]
                    del json_data[col]
            # 如果是空值，就不要有这个key，不然拿到的是控制报错
            for col in copy.deepcopy(expand_value):
                if expand_value[col]==None:
                    del expand_value[col]

            json_data[column_name]=json.dumps(expand_value,indent=4,ensure_ascii=False)
        return json_data

        # @expose("/add", methods=["POST"])
    # def add(self):
    @expose("/", methods=["POST"])
    # @pysnooper.snoop(watch_explode=('item', 'json_data'))
    def api_add(self):
        self.src_item_json = {}
        if not request.is_json:
            return self.response_error(400, message="Request is not JSON")
        try:
            json_data = request.get_json(silent=True)
            for key in json_data:
                if type(json_data[key]) == str:
                    json_data[key] = json_data[key].strip(" ")  # 所有输入去除首尾空格，避免误输入
            if self.pre_add_req:
                json_data_temp = self.pre_add_req(req_json=json_data)
                if json_data_temp:
                    json_data = json_data_temp
            # 将扩展字段先合并在一起
            if self.expand_columns:
                json_data = self.to_expand(json_data)

            json_data = {key: json_data[key] for key in json_data if key in self.add_columns}
            item = self.add_model_schema.load(json_data)
            # item = self.add_model_schema.load(data)
        except Exception as err:
            flash(str(err),'error')
            return self.response_error(422, message=str(err))
        # This validates custom Schema with custom validations
        if isinstance(item, dict):
            return self.response_error(422, message=item.errors)
        try:
            self.pre_add(item)
            self.datamodel.add(item, raise_exception=True)
            self.post_add(item)
            result_data = self.add_model_schema.dump(item, many=False)
            result_data[self.primary_key] = self.datamodel.get_pk_value(item)
            back_data = {
                'result': result_data,
                "status": 0,
                'message': "success"
            }
            return self.response(
                200,
                **back_data,
            )
        except IntegrityError as e:
            return self.response_error(422, message=str(e.orig))
        except Exception as e1:
            return self.response_error(500, message=str(e1))

    @event_logger.log_this
    @expose("/<int:pk>", methods=["PUT"])
    # @pysnooper.snoop(watch_explode=('item','data'))
    def api_edit(self, pk):

        item = self.datamodel.get(pk, self._base_filters)
        self.src_item_json = item.to_json()

        if self.check_edit_permission:
            has_permission = self.check_edit_permission(item)
            if not has_permission:
                flash('no permission to edit', category='warning')
                return self.response_error(422, message='no permission to edit')

        if not request.is_json:
            return self.response_error(400, message="Request is not JSON")
        if not item:
            return self.response_error(404, message='Not found')
        try:
            json_data = request.get_json(silent=True)
            for key in json_data:
                if type(json_data[key]) == str:
                    json_data[key] = json_data[key].strip(" ")  # 所有输入去除首尾空格，避免误输入

            if self.pre_update_req:
                json_data = self.pre_update_req(json_data)

            # 将扩展字段先合并在一起
            if self.expand_columns:
                json_data = self.to_expand(json_data)

            data = self._merge_update_item(item, json_data)
            data = {key: data[key] for key in data if key in self.edit_columns}
            item = self.edit_model_schema.load(data, instance=item)
        except Exception as err:
            return self.response_error(422, message=str(err))
            # return self.response_error(422, message=err.messages)
        # This validates custom Schema with custom validations
        if isinstance(item, dict):
            return self.response_error(422, message=item.errors)
        self.pre_update(item)

        try:
            self.datamodel.edit(item, raise_exception=True)
            if self.post_update:
                self.post_update(item)
            result = self.edit_model_schema.dump(
                item, many=False
            )
            result[self.primary_key] = self.datamodel.get_pk_value(item)
            back_data = {
                "status": 0,
                "message": "success",
                "result": result
            }
            return self.response(
                200,
                **back_data,
            )
        except IntegrityError as e:
            return self.response_error(422, message=str(e.orig))

    @event_logger.log_this
    @expose("/<int:pk>", methods=["DELETE"])
    # @pysnooper.snoop()
    def api_delete(self, pk):
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            return self.response_error(404, message='Not found')

        if self.check_delete_permission:
            has_permission = self.check_delete_permission(item)
            if not has_permission:
                flash('no permission to delete',category='warning')
                return self.response_error(422, message='no permission to delete')

        if self.pre_delete:
            self.pre_delete(item)
        try:
            self.datamodel.delete(item, raise_exception=True)
            self.post_delete(item)
            back_data = {
                "status": 0,
                "message": "success",
                "result": item.to_json()
            }
            return self.response(200, **back_data)
        except IntegrityError as e:
            return self.response_error(422, message=str(e.orig))

    @event_logger.log_this
    @expose("/action/<string:name>/<int:pk>", methods=["GET"])
    def single_action(self, name, pk):
        """
            Action method to handle actions from a show view
        """
        pk = self._deserialize_pk_if_composite(pk)
        action = self.actions.get(name)
        try:
            res = action.func(self.datamodel.get(pk))
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
            return self.response(500, **back)

    @event_logger.log_this
    @expose("/multi_action/<string:name>", methods=["POST"])
    def multi_action(self, name):
        """
            Action method to handle multiple records selected from a list view
        """
        pks = request.json["ids"]
        action = self.actions.get(name)
        items = [
            self.datamodel.get(self._deserialize_pk_if_composite(int(pk))) for pk in pks
        ]
        try:
            back = action.func(items)
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

    @expose("/download_template/", methods=["GET"])
    def download_template(self):

        demostr = ','.join(list(self.add_columns)) + "\n" + ','.join(['xx' for x in list(self.add_columns)])

        file_name = self.datamodel.obj.__tablename__
        csv_file = '%s.csv' % file_name
        if os.path.exists(csv_file):
            os.remove(csv_file)
        file = open(csv_file, mode='w', encoding='utf-8-sig')
        file.writelines(demostr)
        file.close()
        csv_file = os.path.abspath(csv_file)
        response = self.csv_response(csv_file, file_name=file_name)
        return response

    @expose("/echart", methods=["GET"])
    def echart(self):
        _args = request.get_json(silent=True) or {}
        _args.update(json.loads(request.args.get('form_data', "{}")))
        for key in request.args:
            if key!='form_data':
                _args[key.replace('form_data','').replace('[','').replace(']','')]=request.args[key]

        # _args.update(request.args)
        joined_filters = self._handle_filters_args(_args)
        filters = {}
        for flt, value in joined_filters.get_filters_values():
            filters[flt.column_name] = [value] if flt.column_name not in filters else (filters[flt.column_name] + [value])

        if self.echart_option:
            try:
                option = self.echart_option(filters)
                if option:
                    return jsonify({
                        "message": "success",
                        "status": 0,
                        "result": option
                    })
            except Exception as e:
                print(e)

        return jsonify({})

    @expose("/upload/", methods=["POST"])
    def upload(self):
        csv_file = request.files.get('csv_file')  # FileStorage
        # 文件保存至指定路径
        i_path = csv_file.filename
        if os.path.exists(i_path):
            os.remove(i_path)
        csv_file.save(i_path)
        # 读取csv，读取header，按行处理
        import csv
        csv_reader = csv.reader(open(i_path, mode='r', encoding='utf-8-sig'))
        header = None
        result = []
        for line in csv_reader:
            if not header:
                header = line
                continue
            # 判断header里面的字段是否在数据库都有
            for col_name in header:
                # attr = self.datamodel.obj
                if not hasattr(self.datamodel.obj, col_name):
                    message = __('csv首行header与数据库字段不对应')
                    flash(message, 'warning')
                    back = {
                        "status": 1,
                        "result": [],
                        "message": message
                    }
                    return self.response(200, **back)

            # 个数不对的去掉
            if len(line) != len(header):
                continue

            # 全是空值的去掉
            ll = [l.strip() for l in line if l.strip()]
            if not ll:
                continue

            data = dict(zip(header, line))

            try:
                if self.pre_upload:
                    data = self.pre_upload(data)
                model = self.datamodel.obj(**data)
                self.pre_add(model)
                db.session.add(model)
                self.post_add(model)
                db.session.commit()
                result.append('success')
            except Exception as e:
                db.session.rollback()
                print(e)
                result.append(str(e))

        flash('success %s rows，fail %s rows' % (len([x for x in result if x == 'success']), len([x for x in result if x == 'fail'])),'warning')
        back = {
            "status": 0,
            "message": "success %s rows" % len([x for x in result if x == 'success']),
            "result": result
        }
        return self.response(200, **back)

    @expose("/download/", methods=["GET"])
    # @pysnooper.snoop()
    def download(self):
        import pandas
        sqllchemy_uri = conf.get('SQLALCHEMY_DATABASE_URI', '')
        bind_key = getattr(self.datamodel.obj, '__bind_key__', None)
        if bind_key:
            bind_sqllchemy_uri = conf['SQLALCHEMY_BINDS'].get(bind_key, '')
            if bind_sqllchemy_uri:
                sqllchemy_uri = bind_sqllchemy_uri

        import sqlalchemy.engine.url as url
        uri = url.make_url(sqllchemy_uri)
        sql_engine = create_engine(uri)
        table_name = self.datamodel.obj.__tablename__
        # sql = 'select `%s` from %s' % ('`,`'.join(self.show_columns), table_name)
        file_name = '%s.csv' % table_name
        csv_file_path = os.path.abspath(file_name)
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        _args = request.get_json(silent=True) or {}
        req_json = json.loads(request.args.get('form_data', "{}"))  # 根据请求筛选下载数据，而不是全量下载
        _args.update(req_json)
        # 下载搜索出来的
        if _args and _args.get('filters', []):
            joined_filters = self._handle_filters_args(_args)
            count, lst = self.datamodel.query(
                joined_filters,
                '',
                '',
                page=0,
                page_size='20000',
                select_columns=self.show_columns,
            )
            # print(count,lst)
            if count > 0:
                with(open(csv_file_path, 'w', newline='', encoding="utf-8-sig")) as csvfile:
                    csvwrite = csv.writer(csvfile, delimiter=',')
                    csvwrite.writerow(self.show_columns)
                    for item in lst:
                        data = []
                        for col in self.show_columns:
                            data.append(getattr(item, col))
                        csvwrite.writerow(data)
                    csvfile.close()

                response = Response(self.send_file(csv_file_path), content_type='application/octet-stream')
                response.headers["Content-disposition"] = 'attachment; filename=%s' % file_name  # 如果不加上这行代码，导致下图的问题
                return response

        # 下载全量
        sql = 'select * from %s' % (table_name)
        print(sql)
        results = pandas.read_sql_query(sql, sql_engine)
        results.to_csv(csv_file_path, index=False, sep=",", encoding='utf-8-sig')  # index 是第几行的表示

        response = Response(self.send_file(csv_file_path), content_type='application/octet-stream')
        response.headers["Content-disposition"] = 'attachment; filename=%s' % file_name  # 如果不加上这行代码，导致下图的问题
        return response

        # response = self.csv_response(csv_file_path, file_name=table_name)
        # return response

    @expose("/favorite/<pk>", methods=["POST", 'DELETE'])
    # @pysnooper.snoop()
    def favorite(self, pk):
        if request.method == 'POST':
            item = self.datamodel.get(pk, self._base_filters)
            if item:
                exist_favorite = db.session.query(Favorite).filter(Favorite.model_name==self.datamodel.obj.__tablename__).filter(Favorite.row_id==str(pk)).filter(Favorite.user_id==g.user.id).first()
                if not exist_favorite:
                    favorite = Favorite(model_name=self.datamodel.obj.__tablename__, row_id=str(pk), user_id=g.user.id)
                    db.session.add(favorite)
                    db.session.commit()
                back_data = {
                    'result': '',
                    "status": 'success',
                    'message': 'favorite success'
                }
                return self.response(200, **back_data)

            else:
                back_data = {
                    'result': '',
                    "status": 'fail',
                    'message': 'target not exist or no permission'
                }
                return self.response(404, **back_data)
        elif request.method=='DELETE':
            exist_favorite = db.session.query(Favorite).filter(Favorite.model_name == self.datamodel.obj.__tablename__).filter(Favorite.row_id == str(pk)).filter(Favorite.user_id == g.user.id).first()
            if exist_favorite:
                db.session.delete(exist_favorite)
                db.session.commit()
            back_data = {
                'result': '',
                "status": 'success',
                'message': 'unfavorite success'
            }
            return self.response(200, **back_data)

    # @expose("/git/<pk>", methods=["GET"])
    # def git(self,pk):
    #     item = self.datamodel.get(pk, self._base_filters)
    #     return response

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    # @pysnooper.snoop(watch_explode=('items'))
    def muldelete(self, items):
        if not items:
            abort(404)
        success = []
        fail = []
        for item in items:
            try:
                self.pre_delete(item)
                db.session.delete(item)
                success.append(item.to_json())
            except Exception as e:
                flash(str(e), "danger")
                fail.append(item.to_json())
        db.session.commit()
        return json.dumps(
            {
                "success": success,
                "fail": fail
            }, indent=4, ensure_ascii=False
        )

    """
    ------------------------------------------------
                HELPER FUNCTIONS
    ------------------------------------------------
    """

    def _deserialize_pk_if_composite(self, pk):
        def date_deserializer(obj):
            if "_type" not in obj:
                return obj

            from dateutil import parser

            if obj["_type"] == "datetime":
                return parser.parse(obj["value"])
            elif obj["_type"] == "date":
                return parser.parse(obj["value"]).date()
            return obj

        if self.datamodel.is_pk_composite():
            try:
                pk = json.loads(pk, object_hook=date_deserializer)
            except Exception:
                pass
        return pk

    def _handle_page_args(self, rison_args):
        """
            Helper function to handle rison page
            arguments, sets defaults and impose
            FAB_API_MAX_PAGE_SIZE

        :param rison_args:
        :return: (tuple) page, page_size
        """
        page = rison_args.get(API_PAGE_INDEX_RIS_KEY, 0)
        page_size = rison_args.get(API_PAGE_SIZE_RIS_KEY, self.page_size)
        return self._sanitize_page_args(page, page_size)

    # @pysnooper.snoop()
    def _sanitize_page_args(self, page, page_size):
        _page = page or 0
        _page_size = page_size or self.page_size
        max_page_size = self.max_page_size or current_app.config.get(
            "FAB_API_MAX_PAGE_SIZE"
        )
        # Accept special -1 to uncap the page size
        if max_page_size == -1:
            if _page_size == -1:
                return None, None
            else:
                return _page, _page_size
        if _page_size > max_page_size or _page_size < 1:
            _page_size = max_page_size
        return _page, _page_size

    def _handle_order_args(self, rison_args):
        """
            Help function to handle rison order
            arguments

        :param rison_args:
        :return:
        """
        order_column = rison_args.get(API_ORDER_COLUMN_RIS_KEY, "")
        order_direction = rison_args.get(API_ORDER_DIRECTION_RIS_KEY, "")
        if not order_column and self.base_order:
            return self.base_order
        if not order_column:
            return "", ""
        elif order_column not in self.order_columns:
            raise InvalidOrderByColumnFABException(
                f"Invalid order by column: {order_column}"
            )
        return order_column, order_direction

    def _handle_filters_args(self, rison_args):
        self._filters.clear_filters()
        self._filters.rest_add_filters(rison_args.get(API_FILTERS_RIS_KEY, []))
        return self._filters.get_joined_filters(self._base_filters)

    # @pysnooper.snoop(watch_explode=("column"))
    def _description_columns_json(self, cols=None):
        """
            Prepares dict with col descriptions to be JSON serializable
        """
        ret = {}
        cols = cols or []
        d = {k: v for (k, v) in self.description_columns.items() if k in cols}
        for key, value in d.items():
            ret[key] = as_unicode(_(value).encode("UTF-8"))

        edit_form_extra_fields = self.edit_form_extra_fields
        for col in edit_form_extra_fields:
            column = edit_form_extra_fields[col]
            if hasattr(column, 'kwargs') and column.kwargs:
                description = column.kwargs.get('description', '')
                if description:
                    ret[col] = description

        return ret

    # 有可能在过程中变化了label columns
    def _label_columns_json(self, cols=None):
        """
            Prepares dict with labels to be JSON serializable
        """
        ret = {}
        # 自动生成的label
        cols = cols or []
        d = {k: v for (k, v) in self.label_columns.items() if k in cols}
        for key, value in d.items():
            ret[key] = as_unicode(_(value).encode("UTF-8"))

        self._init_label_columns()

        # 扩展字段里面的label
        if self.expand_columns:
            new_columns={}
            [new_columns.update(col) for col in list(self.expand_columns.values())]

            for col_name in new_columns:
                column_field_kwargs = new_columns[col_name].kwargs
                ret[col_name] = column_field_kwargs.get('label',col_name)

        new_label_columns = self.list_columns+self.show_columns+self.edit_columns+self.add_columns
        for col in list(set(new_label_columns)):
            ret[col] = self.label_columns.get(col,col)

        return ret

    def make_ui_info(self, ret_src):
        ret = copy.deepcopy(ret_src)
        # 可序列化处理
        if ret.get('default', None) and isfunction(ret['default']):
            ret['default'] = None  # 函数没法序列化

        # 统一处理校验器
        local_validators = []
        for v in ret.get('validators', []):
            # print(v)
            # if ret.get('name', '') == "warehouse_level":
            #     print(ret)
            #     print(v)
            #     # from wtforms.validators import DataRequired
            #     print(type(v))
            #     print(v.__class__.__name__)
            val = {}
            val['type'] = v.__class__.__name__

            if type(v) == validators.Regexp or type(v) == validate.Regexp:  # 一种是数据库的校验器，一种是可视化的校验器
                val['regex'] = str(v.regex.pattern)
            elif type(v) == validators.Length or type(v) == validate.Length:
                val['min'] = v.min
                val['max'] = v.max
            elif type(v) == validators.NumberRange or type(v) == validate.Range:
                val['min'] = v.min
                val['max'] = v.max
            else:
                pass

            local_validators.append(val)
        ret['validators'] = local_validators

        # 统一规范前端type和选择时value
        # 选择器
        if ret.get('type', '') in ['QuerySelect', 'Select', 'Related', 'MySelectMultiple', 'SelectMultiple', 'Enum']:
            choices = ret.get('choices', [])
            values = ret.get('values', [])
            if choices:
                values = []
                for choice in choices:
                    if choice and len(choice) == 2:
                        values.append({
                            "id": choice[0],
                            "value": choice[1]
                        })
            ret['values'] = values
            if not ret.get('ui-type', ''):
                ret['ui-type'] = 'select2' if 'SelectMultiple' in ret['type'] else 'select'

        # 字符串
        if ret.get('ui-type', '') not in ['list', 'datePicker']:  # list,datePicker 类型，保持原样
            if ret.get('type', '') in ['String', ]:
                if ret.get('widget', 'BS3Text') == 'BS3Text':
                    ret['ui-type'] = 'input'
                else:
                    ret['ui-type'] = 'textArea'

        # 长文本输入
        if 'text' in ret.get('type', '').lower():
            ret['ui-type'] = 'textArea'
        if 'varchar' in ret.get('type', '').lower():
            ret['ui-type'] = 'input'

        # bool类型
        if 'boolean' in ret.get('type', '').lower():
            ret['ui-type'] = 'radio'
            ret['values'] = [
                {
                    "id": True,
                    "value": "yes",
                },
                {
                    "id": False,
                    "value": "no",
                },
            ]
            ret['default'] = True if ret.get('default', 0) else False

        # 处理正则自动输入
        default = ret.get('default', None)
        if default and re.match('\$\{.*\}', str(default)):
            ret['ui-type'] = 'match-input'

        return ret

    # @pysnooper.snoop(watch_explode=('ret'))
    def filed2ui(self,column_name,column_field,ret_src={}):
        ret = copy.deepcopy(ret_src)
        column_field_kwargs = column_field.kwargs
        # type 类型 EnumField   values
        # aa = column_field
        ret['type'] = column_field.field_class.__name__.replace('Field', '')
        # ret['description']=column_field_kwargs.get('description','')
        ret['description'] = self.description_columns.get(column_name, column_field_kwargs.get('description', ''))
        ret['label'] = self.label_columns.get(column_name, column_field_kwargs.get('label', ''))
        ret['default'] = column_field_kwargs.get('default', '')
        ret['validators'] = column_field_kwargs.get('validators', ret.get("validators",[]))
        ret['choices'] = column_field_kwargs.get('choices', [])
        if 'widget' in column_field_kwargs:
            ret['widget'] = column_field_kwargs['widget'].__class__.__name__.replace('Widget', '').replace('Field','').replace('My', '')
            # 处理禁止编辑
            if hasattr(column_field_kwargs['widget'], 'readonly') and column_field_kwargs['widget'].readonly:
                ret['disable'] = True
            # 处理重新拉取info
            if hasattr(column_field_kwargs['widget'], 'retry_info') and column_field_kwargs['widget'].retry_info:
                ret['retry_info'] = True

            # 处理tips
            if hasattr(column_field_kwargs['widget'], 'tips') and column_field_kwargs['widget'].tips:
                ret['tips'] = column_field_kwargs['widget'].tips

            # 处理选填类型
            if hasattr(column_field_kwargs['widget'], 'can_input') and column_field_kwargs['widget'].can_input:
                ret['ui-type'] = 'input-select'

            # 处理时间类型
            if hasattr(column_field_kwargs['widget'], 'is_date') and column_field_kwargs['widget'].is_date:
                ret['ui-type'] = 'datePicker'
            # 处理时间类型
            if hasattr(column_field_kwargs['widget'], 'is_date_range') and column_field_kwargs['widget'].is_date_range:
                ret['ui-type'] = 'rangePicker'

            # 处理扩展字段，一个字段存储一个list的值
            if hasattr(column_field_kwargs['widget'], 'expand_filed') and column_field_kwargs['widget'].expand_filed:
                # print(column_name)
                ret['ui-type'] = 'list'
                ret["info"] = self.columnsfield2info(column_field_kwargs['widget'].expand_filed)

            # 处理内容自动填充可取值，对于那种配置使用过往记录作为可选值的参数进行处理
            if hasattr(column_field_kwargs['widget'], 'conten2choices') and column_field_kwargs['widget'].conten2choices:
                # 先从缓存中拿结果
                field_contents = None
                try:
                    field_contents = cache.get(self.datamodel.obj.__tablename__ + "_" + column_name)
                    # print('read from cache ' + self.datamodel.obj.__tablename__ + "_" + column_name)
                except Exception as e:
                    print(e)
                # 缓存没有数据，再从数据库中读取
                if not field_contents:
                    try:
                        field_contents = db.session.query(getattr(self.datamodel.obj, column_name)).group_by(getattr(self.datamodel.obj, column_name)).all()
                        field_contents = list(set([item[0] for item in field_contents]))
                        cache.set(self.datamodel.obj.__tablename__ + "_" + column_name, field_contents, timeout=60 * 60)

                    except Exception as e:
                        print(e)

                if field_contents:
                    ret['choices'] = [[x, x] for x in list(set(field_contents))]
        return ret

    # @pysnooper.snoop(watch_explode=('field_contents'))
    def _get_field_info(self, field, filter_rel_field, page=None, page_size=None):
        """
            Return a dict with field details
            ready to serve as a response

        :param field: marshmallow field
        :return: dict with field details
        """
        ret = dict()
        ret["name"] = field.name
        # print(ret["name"])
        # print(type(field))
        # print(field)

        # 根据数据库信息添加
        if self.datamodel:
            list_columns = self.datamodel.list_columns  # 只有数据库存储的字段，没有外键字段
            if field.name in list_columns:
                column = list_columns[field.name]
                default = column.default
                # print(type(column.type))
                column_type = column.type
                # aa=column_type
                column_type_str = str(column_type.__class__.__name__)
                if column_type_str == 'Enum':
                    ret['values'] = [
                        {
                            "id": x,
                            "value": x
                        } for x in column.type.enums
                    ]
                # print(column_type)
                # if type(column_type)==
                # print(column.__class__.__name__)
                ret['type'] = column_type_str
                if default:
                    ret['default'] = default.arg

                # print(column)
                # print(column.type)
                # print(type(column))
                # from sqlalchemy.sql.schema import Column
                # # if column
        if field.name in self.remember_columns:
            ret["remember"] = True
        else:
            ret["remember"] = False
        ret["label"] = self.label_columns.get(field.name, "")
        ret["description"] = self.description_columns.get(field.name, "")
        if field.validate and isinstance(field.validate, list):
            ret["validators"] = [v for v in field.validate]
        elif field.validate:
            ret["validators"] = [field.validate]
        else:
            ret["validators"] = []

        # Handles related fields
        if isinstance(field, Related) or isinstance(field, RelatedList):
            ret["count"], ret["values"] = self._get_list_related_field(
                field, filter_rel_field, page=page, page_size=page_size
            )
            ret["validators"].append(validators.DataRequired())

        # 如果是外键，都加必填
        # if

        # 对于非数据库中字段使用字段信息描述类型
        ret["type"] = field.__class__.__name__ if 'type' not in ret else ret["type"]
        ret["required"] = field.required
        ret["unique"] = getattr(field, "unique", False)
        # When using custom marshmallow schemas fields don't have unique property

        # 根据edit_form_extra_fields来确定
        if self.edit_form_extra_fields:
            if field.name in self.edit_form_extra_fields:
                column_field = self.edit_form_extra_fields[field.name]
                ret = self.filed2ui(field.name,column_field,ret)

        # 补充数据库model中定义的是否必填
        columns = [column for column in self.datamodel.obj.__table__._columns if column.name == field.name and hasattr(column, 'nullable') and not column.nullable]
        if columns:
            if 'validators' in ret:
                ret['validators'].append(validators.DataRequired())
            else:
                ret['validators'] = [validators.DataRequired()]

        # print(ret)
        ret = self.make_ui_info(ret)

        return ret

    # @pysnooper.snoop()
    def _get_fields_info(self, cols, model_schema, filter_rel_fields, **kwargs):
        """
            Returns a dict with fields detail
            from a marshmallow schema

        :param cols: list of columns to show info for
        :param model_schema: Marshmallow model schema
        :param filter_rel_fields: expects add_query_rel_fields or
                                    edit_query_rel_fields
        :param kwargs: Receives all rison arguments for pagination
        :return: dict with all fields details
        """
        ret = list()
        for col in cols:
            page = page_size = None
            col_args = kwargs.get(col, {})
            if col_args:
                page = col_args.get(API_PAGE_INDEX_RIS_KEY, None)
                page_size = col_args.get(API_PAGE_SIZE_RIS_KEY, None)

            page_size = 1000
            ret.append(
                self._get_field_info(
                    model_schema.fields[col],
                    filter_rel_fields.get(col, []),
                    page=page,
                    page_size=page_size,
                )
            )
        return ret

    def _get_list_related_field(
            self, field, filter_rel_field, page=None, page_size=None
    ):
        """
            Return a list of values for a related field

        :param field: Marshmallow field
        :param filter_rel_field: Filters for the related field
        :param page: The page index
        :param page_size: The page size
        :return: (int, list) total record count and list of dict with id and value
        """
        ret = list()
        if isinstance(field, Related) or isinstance(field, RelatedList):
            datamodel = self.datamodel.get_related_interface(field.name)
            filters = datamodel.get_filters(datamodel.get_search_columns_list())
            page, page_size = self._sanitize_page_args(page, page_size)
            order_field = self.order_rel_fields.get(field.name)
            if order_field:
                order_column, order_direction = order_field
            else:
                order_column, order_direction = "", ""
            if filter_rel_field:
                filters = filters.add_filter_list(filter_rel_field)
            count, values = datamodel.query(
                filters, order_column, order_direction, page=page, page_size=page_size
            )
            for value in values:
                ret.append({"id": datamodel.get_pk_value(value), "value": str(value)})
        return count, ret

    def _merge_update_item(self, model_item, data):
        """
            Merge a model with a python data structure
            This is useful to turn PUT method into a PATCH also
        :param model_item: SQLA Model
        :param data: python data structure
        :return: python data structure
        """
        data_item = self.edit_model_schema.dump(model_item, many=False)
        for _col in data_item:
            if _col not in data.keys():
                data[_col] = data_item[_col]
        return data
