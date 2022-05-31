import functools
import json
import logging
import re
import traceback
import urllib.parse
from inspect import isfunction
from apispec import yaml_utils
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import ActionItem
from flask import Blueprint, current_app, jsonify, make_response, request
from flask import flash
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
import yaml
from marshmallow import validate
from wtforms import validators
# from wtforms.validators import DataRequired, Regexp, Length, NumberRange
from flask_appbuilder.api.convert import Model2SchemaConverter
from flask_appbuilder.api.schemas import get_info_schema, get_item_schema, get_list_schema
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
    API_PERMISSIONS_RES_KEY,
    API_PERMISSIONS_RIS_KEY,
    API_RESULT_RES_KEY,
    API_SELECT_COLUMNS_RIS_KEY,
    API_SHOW_COLUMNS_RES_KEY,
    API_SHOW_COLUMNS_RIS_KEY,
    API_SHOW_TITLE_RES_KEY,
    API_SHOW_TITLE_RIS_KEY,
    API_URI_RIS_KEY,
    PERMISSION_PREFIX,
)
from flask import (
    abort,
    g
)
from flask_appbuilder.exceptions import FABException, InvalidOrderByColumnFABException
from flask_appbuilder.security.decorators import permission_name, protect,has_access
from flask_appbuilder.api import BaseModelApi,BaseApi,ModelRestApi
from sqlalchemy.sql import sqltypes
from myapp import app, appbuilder,db,event_logger
conf = app.config

log = logging.getLogger(__name__)
API_COLUMNS_INFO_RIS_KEY = 'columns_info'
API_ADD_FIELDSETS_RIS_KEY = 'add_fieldsets'
API_EDIT_FIELDSETS_RIS_KEY = 'edit_fieldsets'
API_SHOW_FIELDSETS_RIS_KEY = 'show_fieldsets'
API_HELP_URL_RIS_KEY = 'help_url'
API_ACTION_RIS_KEY='action'
API_ROUTE_RIS_KEY ='route_base'

API_PERMISSIONS_RIS_KEY="permissions"
API_USER_PERMISSIONS_RIS_KEY="user_permissions"
API_RELATED_RIS_KEY="related"

def get_error_msg():

    if current_app.config.get("FAB_API_SHOW_STACKTRACE"):
        return traceback.format_exc()
    return "Fatal error"


def safe(f):

    def wraps(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except BadRequest as e:
            return self.response_error(400,message=str(e))
        except Exception as e:
            logging.exception(e)
            return self.response_error(500,message=get_error_msg())

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
                            return self.response_error(400,message="Not a valid rison/json argument"
                            )
                    else:
                        return self.response_error(400,message="Not a valid rison argument")
            if schema:
                try:
                    jsonschema.validate(instance=kwargs["rison"], schema=schema)
                except jsonschema.ValidationError as e:
                    return self.response_error(400,message=f"Not a valid rison schema {e}")
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
class MyappModelRestApi(ModelRestApi):

    # 定义主键列
    label_title=''
    primary_key = 'id'
    api_type = 'json'
    allow_browser_login = True
    base_filters = []
    page_size = 100
    src_item_object = None    # 原始model对象
    src_item_json={}    # 原始model对象的json
    check_edit_permission = None
    datamodel=None
    post_list=None
    pre_json_load=None
    edit_form_extra_fields={}
    add_form_extra_fields = {}
    add_fieldsets = []
    edit_fieldsets=[]
    show_fieldsets = []
    pre_add_get=None
    pre_update_get=None
    help_url = None
    pre_show = None
    default_filter={}
    actions = {}
    pre_list=None
    user_permissions = {
        "add": True,
        "edit": True,
        "delete": True,
        "show": True
    }
    add_form_query_rel_fields = {}
    edit_form_query_rel_fields={}
    related_views=[]
    add_more_info=None
    remember_columns=[]
    spec_label_columns={}
    base_permissions=['can_add','can_show','can_edit','can_list','can_delete']
    # def pre_list(self,**kargs):
    #     return


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
        flash_json=[]
        for flash in flashes:
            flash_json.append({
                "type":flash[0],
                "message":flash[1]
            })
        resp.headers["api_flashes"] = json.dumps(flash_json)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp

    def _init_titles(self):
        """
            Init Titles if not defined
        """
        super(ModelRestApi, self)._init_titles()
        class_name = self.datamodel.model_name
        if self.label_title:
            self.list_title = "遍历 " + self.label_title
            self.add_title = "添加 " + self.label_title
            self.edit_title = "编辑 " + self.label_title
            self.show_title = "查看 " + self.label_title

        if not self.list_title:
            self.list_title = "List " + self._prettify_name(class_name)
        if not self.add_title:
            self.add_title = "Add " + self._prettify_name(class_name)
        if not self.edit_title:
            self.edit_title = "Edit " + self._prettify_name(class_name)
        if not self.show_title:
            self.show_title = "Show " + self._prettify_name(class_name)
        self.title = self.list_title

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


        # 初始化label字段
        # 全局的label
        if hasattr(self.datamodel.obj,'label_columns') and self.datamodel.obj.label_columns:
            for col in self.datamodel.obj.label_columns:
                self.label_columns[col] = self.datamodel.obj.label_columns[col]

        # 本view特定的label
        for col in self.spec_label_columns:
            self.label_columns[col] = self.spec_label_columns[col]

        self.primary_key = self.datamodel.get_pk_name()


    def _init_model_schemas(self):
        # Create Marshmalow schemas if one is not specified
        if self.list_model_schema is None:
            self.list_model_schema = self.model2schemaconverter.convert(
                self.list_columns
            )
        if self.add_model_schema is None:
            self.add_model_schema = self.model2schemaconverter.convert(
                self.add_columns, nested=False, enum_dump_by_name=True
            )
        if self.edit_model_schema is None:
            self.edit_model_schema = self.model2schemaconverter.convert(
                list(set(list(self.edit_columns+self.show_columns+self.list_columns))), nested=False, enum_dump_by_name=True
            )
        if self.show_model_schema is None:
            self.show_model_schema = self.model2schemaconverter.convert(
                self.show_columns
            )

    # 每个用户对当前记录的权限，base_permissions 是对所有记录的权限
    def check_item_permissions(self,item):
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

    # add_form_extra_fields  里面的字段要能拿到才对
    # @pysnooper.snoop(watch_explode=())
    def merge_add_field_info(self, response, **kwargs):
        _kwargs = kwargs.get("add_columns", {})
        if self.add_form_query_rel_fields:
            self.add_query_rel_fields = self.add_form_query_rel_fields
        add_columns = self._get_fields_info(
            self.add_columns,
            self.add_model_schema,
            self.add_query_rel_fields,
            **_kwargs,
        )

        response[API_ADD_COLUMNS_RES_KEY]=add_columns


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
        response[API_EDIT_COLUMNS_RES_KEY] = edit_columns


    # @pysnooper.snoop(watch_explode=('edit_columns'))
    def merge_add_fieldsets_info(self, response, **kwargs):
        # if self.pre_add_get:
        #     self.pre_add_get()
        add_fieldsets=[]
        if self.add_fieldsets:
            for group in self.add_fieldsets:
                group_name = group[0]
                group_fieldsets=group[1]
                add_fieldsets.append({
                    "group":group_name,
                    "expanded":group_fieldsets['expanded'],
                    "fields":group_fieldsets['fields']
                })

        response[API_ADD_FIELDSETS_RIS_KEY] = add_fieldsets

    # @pysnooper.snoop(watch_explode=('edit_columns'))
    def merge_edit_fieldsets_info(self, response, **kwargs):
        edit_fieldsets=[]
        if self.edit_fieldsets:
            for group in self.edit_fieldsets:
                group_name = group[0]
                group_fieldsets=group[1]
                edit_fieldsets.append({
                    "group":group_name,
                    "expanded":group_fieldsets['expanded'],
                    "fields":group_fieldsets['fields']
                })
        response[API_EDIT_FIELDSETS_RIS_KEY] = edit_fieldsets

    def merge_show_fieldsets_info(self, response, **kwargs):
        show_fieldsets=[]
        if self.show_fieldsets:
            for group in self.show_fieldsets:
                group_name = group[0]
                group_fieldsets=group[1]
                show_fieldsets.append({
                    "group":group_name,
                    "expanded":group_fieldsets['expanded'],
                    "fields":group_fieldsets['fields']
                })
        response[API_SHOW_FIELDSETS_RIS_KEY] = show_fieldsets

    # @pysnooper.snoop()
    def merge_search_filters(self, response, **kwargs):
        # Get possible search fields and all possible operations
        search_filters = dict()
        dict_filters = self._filters.get_search_filters()
        for col in self.search_columns:
            search_filters[col]={}
            search_filters[col]['filter'] = [
                {"name": as_unicode(flt.name), "operator": flt.arg_name}
                for flt in dict_filters[col]
            ]

            # print(col)
            # print(self.datamodel.list_columns)
            # 对于外键全部可选值返回，或者还需要现场查询(现场查询用哪个字段是个问题)
            if self.datamodel and self.edit_model_schema:   # 根据edit_column 生成的model_schema，编辑里面才会读取外键对象列表
                if col in self.edit_model_schema.fields:

                    field = self.edit_model_schema.fields[col]
                    # print(field)
                    if isinstance(field, Related) or isinstance(field, RelatedList):
                        filter_rel_field = self.edit_query_rel_fields.get(col, [])
                        # 获取外键对象list
                        search_filters[col]["count"], search_filters[col]["values"] = self._get_list_related_field(
                            field, filter_rel_field, page=0, page_size=1000
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
                search_filters[col]['type'] = column_field.field_class.__name__.replace('Field', '').replace('My','')
                search_filters[col]['choices'] = column_field_kwargs.get('choices', [])
                # 选-填 字段在搜索时为填写字段
                search_filters[col]['ui-type'] = 'input' if hasattr(column_field_kwargs.get('widget',{}),'can_input') and column_field_kwargs['widget'].can_input else False

            search_filters[col] = self.make_ui_info(search_filters[col])
            # 多选字段在搜索时为单选字段
            if search_filters[col].get('ui-type','')=='select2':
                search_filters[col]['ui-type']='select'

            search_filters[col]['default']=self.default_filter.get(col,'')
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
        columns_info={}
        for attr in dir(self.datamodel.obj):
            value = getattr(self.datamodel.obj, attr) if hasattr(self.datamodel.obj,attr) else None
            if type(value)==InstrumentedAttribute:
                if type(value.comparator)==ColumnProperty.Comparator:
                    columns_info[value.key]={
                        "type":str(value.comparator.type)
                    }
                if type(value.comparator)==RelationshipProperty.Comparator:
                    columns_info[value.key] = {
                        "type":"Relationship"
                    }
        response[API_COLUMNS_INFO_RIS_KEY] = columns_info

    def merge_help_url_info(self, response, **kwargs):
        response[API_HELP_URL_RIS_KEY] = self.help_url

    # @pysnooper.snoop(watch_explode='aa')
    def merge_action_info(self, response, **kwargs):
        actions_info = {}
        for attr_name in self.actions:
            action = self.actions[attr_name]
            actions_info[action.name] = {
                "name":action.name,
                "text":action.text,
                "confirmation":action.confirmation,
                "icon":action.icon,
                "multiple":action.multiple,
                "single":action.single
            }
        response[API_ACTION_RIS_KEY] = actions_info


    def merge_route_info(self, response, **kwargs):
        response[API_ROUTE_RIS_KEY] = "/"+self.route_base.strip('/')+"/"
        response['primary_key']=self.primary_key
        response['label_title'] = self.label_title or self._prettify_name(self.datamodel.model_name)

    # @pysnooper.snoop(watch_explode=())
    # 添加关联model的字段
    def merge_related_field_info(self, response, **kwargs):
        try:
            add_info={}
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
        response[API_LIST_TITLE_RES_KEY] = self.list_title

    def merge_show_title(self, response, **kwargs):
        response[API_SHOW_TITLE_RES_KEY] = self.show_title



    def merge_more_info(self,response,**kwargs):
        if self.add_more_info:
            response = self.add_more_info(response,**kwargs)


    def response_error(self,code,message='error',status=1,result={}):
        back_data = {
            'result': result,
            "status": status,
            'message': message
        }
        return self.response(code, **back_data)



    @expose("/_info", methods=["GET"])
    @merge_response_func(merge_more_info,'more_info')
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
    @merge_response_func(merge_list_title, API_LIST_TITLE_RIS_KEY)
    @merge_response_func(merge_show_title, API_SHOW_TITLE_RIS_KEY)
    @merge_response_func(merge_add_title, API_ADD_TITLE_RIS_KEY)
    @merge_response_func(merge_edit_title, API_EDIT_TITLE_RIS_KEY)
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
        id = _args.get(self.primary_key,'')
        if id:
            item = self.datamodel.get(id)
            if item and self.pre_update_get:
                try:
                    self.pre_update_get(item)
                except Exception as e:
                    print(e)
            if item and self.check_item_permissions:
                try:
                    self.check_item_permissions(item)
                except Exception as e:
                    print(e)
        elif self.pre_add_get:
            try:
                self.pre_add_get()
            except Exception as e:
                print(e)

        self.set_response_key_mappings(_response, self.api_info, _args, **_args)
        return self.response(200, **_response)


    @expose("/<int:pk>", methods=["GET"])
    # @pysnooper.snoop(depth=4)
    def api_get(self, pk, **kwargs):
        if self.pre_show:
            src_item_object = self.datamodel.get(pk, self._base_filters)
            self.pre_show(src_item_object)

        # from flask_appbuilder.models.sqla.interface import SQLAInterface
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            return self.response_error(404, "Not found")

        _response = dict()
        _args = kwargs.get("rison", {})
        if 'form_data' in request.args:
            _args.update(json.loads(request.args.get('form_data')))

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

        data = _show_model_schema.dump(item, many=False).data
        if int(_args.get('str_related',0)):
            for key in data:
                if type(data[key])==dict:
                    data[key]=str(getattr(item,key))

        _response['data'] = data # item.to_json()
        _response['data'][self.primary_key] = pk

        back = self.pre_get(_response)
        back_data = {
            'result': back['data'] if back else _response['data'],
            "status": 0,
            'message': "success"
        }
        return self.response(200, **back_data)


    @expose("/", methods=["GET"])
    # @pysnooper.snoop(watch_explode=('_response'))
    def api_list(self, **kwargs):
        _response = dict()
        if self.pre_json_load:
            req_json = self.pre_json_load(request.json)
        else:
            try:
                req_json = request.json or {}
            except Exception as e:
                print(e)
                req_json={}

        _args = req_json or {}
        _args.update(request.args)
        # 应对那些get无法传递body的请求，也可以把body放在url里面
        if 'form_data' in request.args:
            _args.update(json.loads(request.args.get('form_data')))

        if self.pre_list:
            self.pre_list(**_args)


        # handle select columns
        select_cols = _args.get(API_SELECT_COLUMNS_RIS_KEY, [])
        _pruned_select_cols = [col for col in select_cols if col in self.list_columns]
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
            from flask_appbuilder.models.sqla.filters import FilterEqualFunction, FilterStartsWith

            joined_filters = self._handle_filters_args(_args)
        except FABException as e:
            return self.response_error(400,message=str(e))
        # handle base order
        try:
            order_column, order_direction = self._handle_order_args(_args)
        except InvalidOrderByColumnFABException as e:
            return self.response_error(400,message=str(e))
        # handle pagination
        page_index, page_size = self._handle_page_args(_args)
        # Make the query
        query_select_columns = _pruned_select_cols or self.list_columns
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
        import marshmallow.marshalling
        # for item in lst:
        #     if self.datamodel.is_relation(item)
            # aa =
            # item.project = 'aaa'

        data = _list_model_schema.dump(lst, many=True).data

        # 把外键换成字符串
        if int(_args.get('str_related',0)):
            for index in range(len(data)):
                for key in data[index]:
                    if type(data[index][key])==dict:
                        data[index][key]=str(getattr(lst[index],key))

        _response['data'] = data  # [item.to_json() for item in lst]
        # _response["ids"] = pks
        _response["count"] = count   # 这个是总个数
        for index in range(len(lst)):

            _response['data'][index][self.primary_key]= getattr(lst[index],self.primary_key)

        try:
            self.pre_get_list(_response)
        except Exception as e:
            print(e)

        back_data = {
            'result': _response,# _response['data']
            "status": 0,
            'message': "success"
        }
        return self.response(200, **back_data)

    # @pysnooper.snoop()
    def json_to_item(self,data):
        class Back:
            pass
        back = Back()
        try:
            item = self.datamodel.obj(**data)
            # for key in data:
            #     if hasattr(item,key):
            #         setattr(item,key,data[key])

            setattr(back,'data',item)
        except Exception as e:
            setattr(back, 'data', data)
            setattr(back, 'errors', str(e))
        return back


    # @expose("/add", methods=["POST"])
    # def add(self):
    @expose("/", methods=["POST"])
    # @pysnooper.snoop(watch_explode=('item', 'data'))
    def api_add(self):
        self.src_item_json = {}
        if not request.is_json:
            return self.response_error(400,message="Request is not JSON")
        try:
            if self.pre_json_load:
                json_data = self.pre_json_load(request.json)
            else:
                json_data = request.json

            item = self.add_model_schema.load(json_data)
            # item = self.add_model_schema.load(data)
        except ValidationError as err:
            return self.response_error(422,message=err.messages)
        # This validates custom Schema with custom validations
        if isinstance(item.data, dict):
            return self.response_error(422,message=item.errors)
        try:
            self.pre_add(item.data)
            self.datamodel.add(item.data, raise_exception=True)
            self.post_add(item.data)
            result_data = self.add_model_schema.dump(
                        item.data, many=False
                    ).data
            result_data[self.primary_key] = self.datamodel.get_pk_value(item.data)
            back_data={
                'result': result_data,
                "status":0,
                'message':"success"
            }
            return self.response(
                200,
                **back_data,
            )
        except IntegrityError as e:
            return self.response_error(422,message=str(e.orig))
        except Exception as e1:
            return self.response_error(500, message=str(e1))


    @expose("/<pk>", methods=["PUT"])
    # @pysnooper.snoop(watch_explode=('item','data'))
    def api_edit(self, pk):

        item = self.datamodel.get(pk, self._base_filters)
        self.src_item_json = item.to_json()

        # if self.check_redirect_list_url:
        try:
            if self.check_edit_permission:
                has_permission = self.check_edit_permission(item)
                if not has_permission:
                    return json_response(message='no permission to edit',status=1,result={})

        except Exception as e:
            print(e)
            return json_response(message='check edit permission'+str(e),status=1,result={})


        if not request.is_json:
            return self.response_error(400, message="Request is not JSON")
        if not item:
            return self.response_error(404,message='Not found')
        try:
            if self.pre_json_load:
                json_data = self.pre_json_load(request.json)
            else:
                json_data = request.json
            data = self._merge_update_item(item, json_data)
            item = self.edit_model_schema.load(data, instance=item)
        except ValidationError as err:
            return self.response_error(422,message=err.messages)
        # This validates custom Schema with custom validations
        if isinstance(item.data, dict):
            return self.response_error(422,message=item.errors)
        self.pre_update(item.data)


        try:
            self.datamodel.edit(item.data, raise_exception=True)
            self.post_update(item.data)
            result = self.edit_model_schema.dump(
                item.data, many=False
            ).data
            result[self.primary_key] = self.datamodel.get_pk_value(item.data)
            back_data={
                "status":0,
                "message":"success",
                "result":result
            }
            return self.response(
                200,
                **back_data,
            )
        except IntegrityError as e:
            return self.response_error(422,message=str(e.orig))

    @expose("/<pk>", methods=["DELETE"])
    # @pysnooper.snoop()
    def api_delete(self, pk):
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            return self.response_error(404,message='Not found')
        self.pre_delete(item)
        try:
            self.datamodel.delete(item, raise_exception=True)
            self.post_delete(item)
            back_data={
                "status":0,
                "message":"success",
                "result":item.to_json()
            }
            return self.response(200, **back_data)
        except IntegrityError as e:
            return self.response_error(422,message=str(e.orig))



    @expose("/action/<string:name>/<pk>", methods=["GET"])
    def action(self, name, pk):
        """
            Action method to handle actions from a show view
        """
        pk = self._deserialize_pk_if_composite(pk)
        action = self.actions.get(name)
        try:
            res = action.func(self.datamodel.get(pk))
            return jsonify({
                "status": 0,
                "result": {},
                "message": 'success'

            })
        except Exception as e:
            print(e)
            return jsonify({
                "status": -1,
                "message": str(e),
                "result": {}
            })


    @expose("/multi_action/<string:name>", methods=["POST"])
    def multi_action(self,name):
        """
            Action method to handle multiple records selected from a list view
        """
        pks = request.json["ids"]
        action = self.actions.get(name)
        items = [
            self.datamodel.get(self._deserialize_pk_if_composite(pk)) for pk in pks
        ]
        try:
            back = action.func(items)
            message = back if type(back)==str else 'success'
            return jsonify({
                "status":0,
                "result":{},
                "message":message
            })
        except Exception as e:
            print(e)
            return jsonify({
                "status":-1,
                "message": str(e),
                "result":{}
            })




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
                description = column.kwargs.get('description','')
                if description:
                    ret[col] = description

        return ret


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

        # 全局的label
        if hasattr(self.datamodel.obj,'label_columns') and self.datamodel.obj.label_columns:
            for col in self.datamodel.obj.label_columns:
                ret[col] = self.datamodel.obj.label_columns[col]

        # 本view特定的label
        for col in self.spec_label_columns:
            ret[col] = self.spec_label_columns[col]

        return ret

    def make_ui_info(self,ret):

        # 可序列化处理
        if ret.get('default',None) and isfunction(ret['default']):
            ret['default'] = None  # 函数没法序列化


        # print(ret)

        # 统一处理校验器
        local_validators=[]
        for v in ret.get('validators',[]):
            # print(type(v))
            val = {}
            val['type'] = v.__class__.__name__
            if type(v) == validators.Regexp or type(v) == validate.Regexp:
                val['regex'] = str(v.regex.pattern)
            elif type(v) == validators.Length or type(v) == validate.Length:
                val['min'] = v.min
                val['max'] = v.max
            elif type(v) == validators.NumberRange or type(v) == validate.Range :
                val['min'] = v.min
                val['max'] = v.max
            else:
                pass

            local_validators.append(val)
        ret['validators']=local_validators

        # 统一规范前端type和选择时value
        # 选择器
        if ret.get('type','') in ['QuerySelect','Select','Related','MySelectMultiple','SelectMultiple','Enum']:
            choices = ret.get('choices',[])
            values = ret.get('values',[])
            for choice in choices:
                if len(choice)==2:
                    values.append({
                        "id":choice[0],
                        "value":choice[1]
                    })
            ret['values']=values
            if not ret.get('ui-type',''):
                ret['ui-type']='select2' if 'SelectMultiple' in ret['type'] else 'select'

        # 字符串
        if ret.get('type','') in ['String',]:
            if ret.get('widget','BS3Text')=='BS3Text':
                ret['ui-type'] = 'input'
            else:
                ret['ui-type'] = 'textArea'
        # 长文本输入
        if 'text' in ret.get('type','').lower():
            ret['ui-type'] = 'textArea'
        if 'varchar' in ret.get('type','').lower():
            ret['ui-type'] = 'input'

        # bool类型
        if 'boolean' in ret.get('type','').lower():
            ret['ui-type'] = 'radio'
            ret['values']=[
                {
                    "id":True,
                    "value":"yes",
                },
                {
                    "id": False,
                    "value": "no",
                },
            ]
            ret['default']=True if ret.get('default',0) else False

        # 处理正则自动输入
        default = ret.get('default',None)
        if default and re.match('\$\{.*\}',str(default)):
            ret['ui-type']='match-input'

        return ret

    # @pysnooper.snoop(watch_explode=('column','aa'))
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
            list_columns = self.datamodel.list_columns     # 只有数据库存储的字段，没有外键字段
            if field.name in list_columns:
                column = list_columns[field.name]
                default = column.default
                # print(type(column.type))
                column_type=column.type
                # aa=column_type
                column_type_str = str(column_type.__class__.__name__)
                if column_type_str=='Enum':
                    ret['values']=[
                        {
                            "id":x,
                            "value":x
                        } for x in column.type.enums
                    ]
                # print(column_type)
                # if type(column_type)==
                # print(column.__class__.__name__)
                from sqlalchemy.sql.schema import Column
                ret['type']=column_type_str
                if default:
                    ret['default'] = default.arg

                # print(column)
                # print(column.type)
                # print(type(column))
                # from sqlalchemy.sql.schema import Column
                # # if column
        if field.name in self.remember_columns:
            ret["remember"]=True
        else:
            ret["remember"] = False
        ret["label"] = _(self.label_columns.get(field.name, ""))
        ret["description"] = _(self.description_columns.get(field.name, ""))
        # Handles related fields
        if isinstance(field, Related) or isinstance(field, RelatedList):
            ret["count"], ret["values"] = self._get_list_related_field(
                field, filter_rel_field, page=page, page_size=page_size
            )

        if field.validate and isinstance(field.validate, list):
            ret["validators"] = [v for v in field.validate]
        elif field.validate:
            ret["validators"] = [field.validate]

        # 对于非数据库中字段使用字段信息描述类型
        ret["type"] = field.__class__.__name__ if 'type' not in ret else ret["type"]
        ret["required"] = field.required
        ret["unique"] = getattr(field, "unique", False)
        # When using custom marshmallow schemas fields don't have unique property


        # 根据edit_form_extra_fields来确定
        if self.edit_form_extra_fields:
            if field.name in self.edit_form_extra_fields:
                column_field = self.edit_form_extra_fields[field.name]
                column_field_kwargs = column_field.kwargs
                # type 类型 EnumField   values
                # aa = column_field
                ret['type']=column_field.field_class.__name__.replace('Field','')
                # ret['description']=column_field_kwargs.get('description','')
                ret['description'] = self.description_columns.get(field.name,column_field_kwargs.get('description', ''))
                ret['label'] = self.label_columns.get(field.name,column_field_kwargs.get('label', ''))
                ret['default'] = column_field_kwargs.get('default', '')
                ret['validators'] = column_field_kwargs.get('validators', [])
                ret['choices'] = column_field_kwargs.get('choices', [])
                if 'widget' in column_field_kwargs:
                    ret['widget']=column_field_kwargs['widget'].__class__.__name__.replace('Widget','').replace('Field','').replace('My','')
                    ret['disable']=column_field_kwargs['widget'].readonly if hasattr(column_field_kwargs['widget'],'readonly') else False
                    # if hasattr(column_field_kwargs['widget'],'can_input'):
                    #     print(field.name,column_field_kwargs['widget'].can_input)
                    ret['ui-type'] = 'input-select' if hasattr(column_field_kwargs['widget'],'can_input') and column_field_kwargs['widget'].can_input else False


        # print(ret)
        ret=self.make_ui_info(ret)

        return ret

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
            page_size=1000
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
        data_item = self.edit_model_schema.dump(model_item, many=False).data
        for _col in self.edit_columns:
            if _col not in data.keys():
                data[_col] = data_item[_col]
        return data

