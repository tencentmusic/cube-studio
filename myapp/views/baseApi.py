import functools
import json
import logging
import re
import traceback
import urllib.parse

from apispec import yaml_utils
from flask import Blueprint, current_app, jsonify, make_response, request
from flask_babel import lazy_gettext as _
import jsonschema
from marshmallow import ValidationError
from marshmallow_sqlalchemy.fields import Related, RelatedList
import prison
from sqlalchemy.exc import IntegrityError
from werkzeug.exceptions import BadRequest
import yaml

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
from flask_appbuilder.exceptions import FABException, InvalidOrderByColumnFABException
from flask_appbuilder.security.decorators import permission_name, protect,has_access
from flask_appbuilder.api import BaseModelApi,BaseApi,ModelRestApi

from myapp import app, appbuilder,db,event_logger
conf = app.config

log = logging.getLogger(__name__)


def get_error_msg():
    """
        (inspired on Superset code)
    :return: (str)
    """
    if current_app.config.get("FAB_API_SHOW_STACKTRACE"):
        return traceback.format_exc()
    return "Fatal error"


def safe(f):
    """
    A decorator that catches uncaught exceptions and
    return the response in JSON format (inspired on Superset code)
    """
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

    # @pysnooper.snoop()
    def merge_add_field_info(self, response, **kwargs):
        _kwargs = kwargs.get("add_columns", {})
        response[API_ADD_COLUMNS_RES_KEY] = self._get_fields_info(
            self.add_columns,
            self.add_model_schema,
            self.add_query_rel_fields,
            **_kwargs,
        )

    def merge_edit_field_info(self, response, **kwargs):
        _kwargs = kwargs.get("edit_columns", {})
        response[API_EDIT_COLUMNS_RES_KEY] = self._get_fields_info(
            self.edit_columns,
            self.edit_model_schema,
            self.edit_query_rel_fields,
            **_kwargs,
        )

    def merge_search_filters(self, response, **kwargs):
        # Get possible search fields and all possible operations
        search_filters = dict()
        dict_filters = self._filters.get_search_filters()
        for col in self.search_columns:
            search_filters[col] = [
                {"name": as_unicode(flt.name), "operator": flt.arg_name}
                for flt in dict_filters[col]
            ]
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

    def merge_list_title(self, response, **kwargs):
        response[API_LIST_TITLE_RES_KEY] = self.list_title

    def merge_show_title(self, response, **kwargs):
        response[API_SHOW_TITLE_RES_KEY] = self.show_title


    def response_error(self,code,message='error',status=1,result={}):
        back_data = {
            'result': result,
            "status": status,
            'message': message
        }
        return self.response(code, **back_data)


    @expose("/_info", methods=["GET"])
    @merge_response_func(BaseApi.merge_current_user_permissions, API_PERMISSIONS_RIS_KEY)
    @merge_response_func(merge_add_field_info, API_ADD_COLUMNS_RIS_KEY)
    @merge_response_func(merge_edit_field_info, API_EDIT_COLUMNS_RIS_KEY)
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
    def api_info(self, **kwargs):
        _response = dict()
        _args = kwargs.get("rison", {})
        self.set_response_key_mappings(_response, self.api_info, _args, **_args)
        return self.response(200, **_response)


    @expose("/<int:pk>", methods=["GET"])
    # @pysnooper.snoop()
    def api_get(self, pk, **kwargs):
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            return self.response_error(404, "Not found")

        _response = dict()
        _args = kwargs.get("rison", {})
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
        _response['data'] = _show_model_schema.dump(item, many=False).data # item.to_json()
        _response['data']["id"] = pk

        self.pre_get(_response)
        back_data = {
            'result': _response['data'],
            "status": 0,
            'message': "success"
        }
        return self.response(200, **back_data)


    @expose("/", methods=["GET"])
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
        _response['data'] = _list_model_schema.dump(lst, many=True).data  # [item.to_json() for item in lst]
        # _response["ids"] = pks
        _response["count"] = count   # 这个是总个数
        for index in range(len(lst)):
            _response['data'][index]['id']= lst[index].id

        self.pre_get_list(_response)
        back_data = {
            'result': _response['data'],
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
                json = self.pre_json_load(request.json)
            else:
                json = request.json

            item = self.add_model_schema.load(json)
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
            result_data['id'] = self.datamodel.get_pk_value(item.data)
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
                json = self.pre_json_load(request.json)
            else:
                json = request.json
            data = self._merge_update_item(item, json)
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
            result['id'] = self.datamodel.get_pk_value(item.data)
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



    """
    ------------------------------------------------
                HELPER FUNCTIONS
    ------------------------------------------------
    """

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
        cols = cols or []
        d = {k: v for (k, v) in self.label_columns.items() if k in cols}
        for key, value in d.items():
            ret[key] = as_unicode(_(value).encode("UTF-8"))

        if hasattr(self.datamodel.obj,'label_columns') and self.datamodel.obj.label_columns:
            for col in self.datamodel.obj.label_columns:
                ret[col] = self.datamodel.obj.label_columns[col]

        return ret

    # @pysnooper.snoop(watch_explode=("field",'datamodel','column','default'))
    def _get_field_info(self, field, filter_rel_field, page=None, page_size=None):
        """
            Return a dict with field details
            ready to serve as a response

        :param field: marshmallow field
        :return: dict with field details
        """
        ret = dict()
        ret["name"] = field.name
        if self.datamodel:
            list_columns = self.datamodel.list_columns
            if field.name in list_columns:
                column = list_columns[field.name]
                default = column.default
                if default:
                    ret['default']=default.arg
        ret["label"] = _(self.label_columns.get(field.name, ""))

        ret["description"] = _(self.description_columns.get(field.name, ""))
        # if field.name in self.edit_form_extra_fields:
        #     if hasattr(self.edit_form_extra_fields[field.name],'label'):
        #         ret["label"] = self.edit_form_extra_fields[field.name].label
        #     if hasattr(self.edit_form_extra_fields[field.name], 'description'):
        #         ret["description"] = self.edit_form_extra_fields[field.name].description

        # Handles related fields
        if isinstance(field, Related) or isinstance(field, RelatedList):
            ret["count"], ret["values"] = self._get_list_related_field(
                field, filter_rel_field, page=page, page_size=page_size
            )
        if field.validate and isinstance(field.validate, list):
            ret["validate"] = [str(v) for v in field.validate]
        elif field.validate:
            ret["validate"] = [str(field.validate)]
        ret["type"] = field.__class__.__name__
        ret["required"] = field.required
        # When using custom marshmallow schemas fields don't have unique property
        ret["unique"] = getattr(field, "unique", False)
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
