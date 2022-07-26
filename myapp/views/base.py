
import datetime
import os
import functools
import logging
import traceback
from typing import Any, Dict
import pysnooper
from flask_appbuilder.forms import GeneralModelConverter
from flask import abort, flash, g, get_flashed_messages, redirect, Response
from flask_appbuilder import BaseView, ModelView,urltools
from flask_appbuilder.actions import action
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.models.sqla.filters import BaseFilter
from flask_appbuilder.widgets import ListWidget
from myapp.forms import MySearchWidget
from flask_babel import get_locale
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_wtf.form import FlaskForm
import simplejson as json
from werkzeug.exceptions import HTTPException
from wtforms.fields.core import Field, UnboundField
from flask_appbuilder import ModelView, ModelRestApi
import yaml
from flask_appbuilder.security.decorators import has_access, has_access_api, permission_name
from flask_appbuilder.baseviews import BaseCRUDView, BaseFormView, BaseView, expose, expose_api
from myapp import conf, db, get_feature_flags, security_manager,event_logger
from myapp.exceptions import MyappException, MyappSecurityException
from myapp.translations.utils import get_language_pack
from myapp.utils import core
from sqlalchemy import or_
from flask_appbuilder.urltools import (
    get_filter_args,
    get_order_args,
    get_page_args,
    get_page_size_args,
    Stack,
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
from flask import Flask, jsonify

from apispec import yaml_utils
from flask import Blueprint, current_app, jsonify, make_response, request
from flask_babel import lazy_gettext as _

import yaml

FRONTEND_CONF_KEYS = (
    "MYAPP_WEBSERVER_TIMEOUT",
    "ENABLE_JAVASCRIPT_CONTROLS",
    "MYAPP_WEBSERVER_DOMAINS",
)



from flask_appbuilder.const import (
    FLAMSG_ERR_SEC_ACCESS_DENIED,
    LOGMSG_ERR_SEC_ACCESS_DENIED,
    PERMISSION_PREFIX
)
from flask_appbuilder._compat import as_unicode

log = logging.getLogger(__name__)

def has_access(f):
    """
        Use this decorator to enable granular security permissions to your methods.
        Permissions will be associated to a role, and roles are associated to users.

        By default the permission's name is the methods name.
    """
    if hasattr(f, '_permission_name'):
        permission_str = f._permission_name
    else:
        permission_str = f.__name__

    def wraps(self, *args, **kwargs):

        permission_str = "{}{}".format(PERMISSION_PREFIX, f._permission_name)
        if self.method_permission_name:
            _permission_name = self.method_permission_name.get(f.__name__)
            if _permission_name:
                permission_str = "{}{}".format(PERMISSION_PREFIX, _permission_name)
        if (permission_str in self.base_permissions and
                self.appbuilder.sm.has_access(
                    permission_str,
                    self.class_permission_name
                )):
            return f(self, *args, **kwargs)
        else:
            log.warning(
                LOGMSG_ERR_SEC_ACCESS_DENIED.format(
                    permission_str,
                    self.__class__.__name__
                )
            )
            flash(as_unicode(FLAMSG_ERR_SEC_ACCESS_DENIED), "danger")
        return redirect(
            url_for(
                self.appbuilder.sm.auth_view.__class__.__name__ + ".login",
                next=request.url
            )
        )

    f._permission_name = permission_str
    return functools.update_wrapper(wraps, f)


def has_access_api(f):
    """
        Use this decorator to enable granular security permissions to your API methods.
        Permissions will be associated to a role, and roles are associated to users.

        By default the permission's name is the methods name.

        this will return a message and HTTP 401 is case of unauthorized access.
    """
    if hasattr(f, '_permission_name'):
        permission_str = f._permission_name
    else:
        permission_str = f.__name__

    def wraps(self, *args, **kwargs):
        permission_str = "{}{}".format(PERMISSION_PREFIX, f._permission_name)
        if self.method_permission_name:
            _permission_name = self.method_permission_name.get(f.__name__)
            if _permission_name:
                permission_str = "{}{}".format(PERMISSION_PREFIX, _permission_name)
        if (permission_str in self.base_permissions and
                self.appbuilder.sm.has_access(
                    permission_str,
                    self.class_permission_name
                )):
            return f(self, *args, **kwargs)
        else:
            log.warning(
                LOGMSG_ERR_SEC_ACCESS_DENIED.format(
                    permission_str,
                    self.__class__.__name__
                )
            )
            response = make_response(
                jsonify(
                    {
                        'message': str(FLAMSG_ERR_SEC_ACCESS_DENIED),
                        'severity': 'danger'
                    }
                ),
                401
            )
            response.headers['Content-Type'] = "application/json"
            return response

    f._permission_name = permission_str
    return functools.update_wrapper(wraps, f)


def get_error_msg():
    if conf.get("SHOW_STACKTRACE"):
        error_msg = traceback.format_exc()
    else:
        error_msg = "FATAL ERROR \n"
        error_msg += (
            "Stacktrace is hidden. Change the SHOW_STACKTRACE "
            "configuration setting to enable it"
        )
    return error_msg


def json_error_response(msg=None, status=500, stacktrace=None, payload=None, link=None):
    if not payload:
        payload = {"error": "{}".format(msg)}
        payload["stacktrace"] = core.get_stacktrace()
    if link:
        payload["link"] = link

    return Response(
        json.dumps(payload, default=core.json_iso_dttm_ser, ignore_nan=True),
        status=status,
        mimetype="application/json",
    )

def json_response(message,status,result):
    return jsonify(
        {
            "message":message,
            "status":status,
            "result":result
        }
    )

def json_success(json_msg, status=200):
    return Response(json_msg, status=status, mimetype="application/json")


def data_payload_response(payload_json, has_error=False):
    status = 400 if has_error else 200
    return json_success(payload_json, status=status)

# 产生下载csv的响应header
def generate_download_headers(extension, filename=None):
    filename = filename if filename else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    content_disp = "attachment; filename={}.{}".format(filename, extension)
    headers = {"Content-Disposition": content_disp}
    return headers


def api(f):
    """
    A decorator to label an endpoint as an API. Catches uncaught exceptions and
    return the response in the JSON format
    """

    def wraps(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            logging.exception(e)
            return json_error_response(get_error_msg())

    return functools.update_wrapper(wraps, f)


def handle_api_exception(f):
    """
    A decorator to catch myapp exceptions. Use it after the @api decorator above
    so myapp exception handler is triggered before the handler for generic
    exceptions.
    """

    def wraps(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except MyappSecurityException as e:
            logging.exception(e)
            return json_error_response(
                core.error_msg_from_exception(e),
                status=e.status,
                stacktrace=core.get_stacktrace(),
                link=e.link,
            )
        except MyappException as e:
            logging.exception(e)
            return json_error_response(
                core.error_msg_from_exception(e),
                stacktrace=core.get_stacktrace(),
                status=e.status,
            )
        except HTTPException as e:
            logging.exception(e)
            return json_error_response(
                core.error_msg_from_exception(e),
                stacktrace=traceback.format_exc(),
                status=e.code,
            )
        except Exception as e:
            logging.exception(e)
            return json_error_response(
                core.error_msg_from_exception(e), stacktrace=core.get_stacktrace()
            )

    return functools.update_wrapper(wraps, f)

# 获取用户的角色
def get_user_roles():
    if g.user.is_anonymous:
        public_role = conf.get("AUTH_ROLE_PUBLIC")
        return [security_manager.find_role(public_role)] if public_role else []
    return g.user.roles


class BaseMyappView(BaseView):
    # json响应
    def json_response(self, obj, status=200):
        return Response(
            json.dumps(obj, default=core.json_int_dttm_ser, ignore_nan=True),
            status=status,
            mimetype="application/json",
        )
    # 前端显示数据
    def common_bootstrap_payload(self):
        """Common data always sent to the client"""
        messages = get_flashed_messages(with_categories=True)
        locale = str(get_locale())
        return {
            "flash_messages": messages,
            "conf": {k: conf.get(k) for k in FRONTEND_CONF_KEYS},
            "locale": locale,
            "language_pack": get_language_pack(locale),
            "feature_flags": get_feature_flags(),
        }


# 自定义list页面
class MyappListWidget(ListWidget):
    template = "myapp/fab_overrides/list.html"


# model 页面基本视图
class MyappModelView(ModelView):
    api_type='web'
    datamodel=None
    page_size = 100
    list_widget = MyappListWidget
    src_item_object = None    # 原始model对象
    src_item_json={}    # 原始model对象的json
    check_redirect_list_url=None
    search_widget = MySearchWidget
    help_url=''

    pre_add_get = None
    pre_update_get = None
    post_list = None
    pre_show = None
    post_show = None
    check_edit_permission=None
    label_title = ''

    conv = GeneralModelConverter(datamodel)
    pre_list=None
    user_permissions = {
        "can_add": True,
        "can_edit": True,
        "can_delete": True,
        "can_show": True
    }


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
            flash_json.append([flash[0],flash[1]])
        resp.headers["api_flashes"] = json.dumps(flash_json)
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp



    # 配置增删改查页面标题
    def _init_titles(self):

        self.help_url = conf.get('HELP_URL', {}).get(self.datamodel.obj.__tablename__, '') if self.datamodel else ''

        """
            Init Titles if not defined
        """
        class_name = self.datamodel.model_name
        if not self.list_title:
            if not self.label_title:
                self.list_title = "List " + self._prettify_name(class_name)
            else:
                self.list_title  = self.label_title + " 列表"
        if not self.add_title:
            if not self.label_title:
                self.add_title = "Add " + self._prettify_name(class_name)
            else:
                self.add_title = '添加 ' + self.label_title
        if not self.edit_title:
            if not self.label_title:
                self.edit_title = "Edit " + self._prettify_name(class_name)
            else:
                self.edit_title ='修改 ' + self.label_title
        if not self.show_title:
            if not self.label_title:
                self.show_title = "Show " + self._prettify_name(class_name)
            else:
                self.show_title = self.label_title+" 详情"
        self.title = self.list_title

    # 每个用户对当前记录的权限，base_permissions 是对所有记录的权限
    def check_item_permissions(self,item):
        self.user_permissions = {
            "add": True,
            "edit": True,
            "delete": True,
            "show": True
        }

    # 配置字段的中文描述
    # @pysnooper.snoop()
    def _gen_labels_columns(self, list_columns):
        """
            Auto generates pretty label_columns from list of columns
        """
        if hasattr(self.datamodel.obj,'label_columns') and self.datamodel.obj.label_columns:
            for col in self.datamodel.obj.label_columns:
                self.label_columns[col] = self.datamodel.obj.label_columns[col]

        for col in list_columns:
            if not self.label_columns.get(col):
                self.label_columns[col] = self._prettify_column(col)

    # 获取列的中文显示
    def lab(self,col):
        if col in self.label_columns:
            return _(self.label_columns[col])
        return _(self._prettify_column(col))

    def pre_delete(self, item):
        pass

    def _get_search_widget(self, form=None, exclude_cols=None, widgets=None):
        exclude_cols = exclude_cols or []
        widgets = widgets or {}
        widgets["search"] = self.search_widget(
            route_base=self.route_base,
            form=form,
            include_cols=self.search_columns,
            exclude_cols=exclude_cols,
            filters=self._filters,
            help_url = self.help_url
        )
        return widgets


    def _get_list_widget(
        self,
        filters,
        actions=None,
        order_column="",
        order_direction="",
        page=None,
        page_size=None,
        widgets=None,
        **args,
    ):

        """ get joined base filter and current active filter for query """
        widgets = widgets or {}
        actions = actions or self.actions
        page_size = page_size or self.page_size
        if not order_column and self.base_order:
            order_column, order_direction = self.base_order
        joined_filters = filters.get_joined_filters(self._base_filters)
        count, lst = self.datamodel.query(
            joined_filters,
            order_column,
            order_direction,
            page=page,
            page_size=page_size,
        )
        if self.post_list:
            lst = self.post_list(lst)
        pks = self.datamodel.get_keys(lst)

        # serialize composite pks
        pks = [self._serialize_pk_if_composite(pk) for pk in pks]

        widgets["list"] = self.list_widget(
            label_columns=self.label_columns,
            include_columns=self.list_columns,
            value_columns=self.datamodel.get_values(lst, self.list_columns),
            order_columns=self.order_columns,
            formatters_columns=self.formatters_columns,
            page=page,
            page_size=page_size,
            count=count,
            pks=pks,
            actions=actions,
            filters=filters,
            modelview_name=self.__class__.__name__,
        )
        return widgets


    @event_logger.log_this
    @expose("/list/")
    @has_access
    def list(self):
        if self.pre_list:
            self.pre_list()
        widgets = self._list()
        res = self.render_template(
            self.list_template, title=self.list_title, widgets=widgets
        )
        return res


    @event_logger.log_this
    @expose("/show/<pk>", methods=["GET"])
    @has_access
    def show(self, pk):
        pk = self._deserialize_pk_if_composite(pk)

        if self.pre_show:
            src_item_object = self.datamodel.get(pk, self._base_filters)
            self.pre_show(src_item_object)
        widgets = self._show(pk)
        return self.render_template(
            self.show_template,
            pk=pk,
            title=self.show_title,
            widgets=widgets,
            related_views=self._related_views,
        )


    # @pysnooper.snoop(watch_explode=('item'))
    def _add(self):
        """
            Add function logic, override to implement different logic
            returns add widget or None
        """
        is_valid_form = True
        get_filter_args(self._filters)
        exclude_cols = self._filters.get_relation_cols()
        form = self.add_form.refresh()

        if request.method == "POST":
            self._fill_form_exclude_cols(exclude_cols, form)
            if form.validate():
                self.process_form(form, True)
                item = self.datamodel.obj()

                try:
                    form.populate_obj(item)
                    self.pre_add(item)
                except Exception as e:
                    flash(str(e), "danger")
                else:
                    print(item.to_json())
                    if self.datamodel.add(item):
                        self.post_add(item)
                    flash(*self.datamodel.message)
                finally:
                    return None
            else:
                is_valid_form = False
        if is_valid_form:
            self.update_redirect()
        return self._get_add_widget(form=form, exclude_cols=exclude_cols)


    @event_logger.log_this
    @expose("/add", methods=["GET", "POST"])
    @has_access
    def add(self):
        self.src_item_json = {}
        if request.method=='GET' and self.pre_add_get:
            try:
                self.pre_add_get()
                self.conv = GeneralModelConverter(self.datamodel)
                self.add_form = self.conv.create_form(
                    self.label_columns,
                    self.add_columns,
                    self.description_columns,
                    self.validators_columns,
                    self.add_form_extra_fields,
                    self.add_form_query_rel_fields,
                )
            except Exception as e:
                print(e)
                return redirect(self.get_redirect())

        widget = self._add()
        if not widget:
            if self.check_redirect_list_url:
                return redirect(self.check_redirect_list_url)
            return self.post_add_redirect()
        else:
            return self.render_template(
                self.add_template, title=self.add_title, widgets=widget
            )



    # @pysnooper.snoop(watch_explode=('item'))
    def _edit(self, pk):
        """
            Edit function logic, override to implement different logic
            returns Edit widget and related list or None
        """
        is_valid_form = True
        pages = get_page_args()
        page_sizes = get_page_size_args()
        orders = get_order_args()
        get_filter_args(self._filters)
        exclude_cols = self._filters.get_relation_cols()

        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            abort(404)
        # convert pk to correct type, if pk is non string type.
        pk = self.datamodel.get_pk_value(item)

        if request.method == "POST":
            form = self.edit_form.refresh(request.form)

            # fill the form with the suppressed cols, generated from exclude_cols
            self._fill_form_exclude_cols(exclude_cols, form)
            # trick to pass unique validation
            form._id = pk
            if form.validate():
                self.process_form(form, False)

                try:
                    form.populate_obj(item)
                    self.pre_update(item)
                except Exception as e:
                    flash(str(e), "danger")
                else:
                    if self.datamodel.edit(item):
                        self.post_update(item)
                    flash(*self.datamodel.message)
                finally:
                    return None
            else:
                is_valid_form = False
        else:
            # Only force form refresh for select cascade events
            form = self.edit_form.refresh(obj=item)
            # Perform additional actions to pre-fill the edit form.
            self.prefill_form(form, pk)

        widgets = self._get_edit_widget(form=form, exclude_cols=exclude_cols)
        widgets = self._get_related_views_widgets(
            item,
            filters={},
            orders=orders,
            pages=pages,
            page_sizes=page_sizes,
            widgets=widgets,
        )
        if is_valid_form:
            self.update_redirect()
        return widgets

    @event_logger.log_this
    @expose("/edit/<pk>", methods=["GET", "POST"])
    @has_access
    def edit(self, pk):
        pk = self._deserialize_pk_if_composite(pk)
        self.src_item_object = self.datamodel.get(pk, self._base_filters)

        if request.method=='GET' and self.pre_update_get and self.src_item_object:
            try:
                self.pre_update_get(self.src_item_object)
                self.conv = GeneralModelConverter(self.datamodel)
                # 重新更新，而不是只在初始化时更新
                self.edit_form = self.conv.create_form(
                    self.label_columns,
                    self.edit_columns,
                    self.description_columns,
                    self.validators_columns,
                    self.edit_form_extra_fields,
                    self.edit_form_query_rel_fields,
                )
            except Exception as e:
                print(e)
                self.update_redirect()
                return redirect(self.get_redirect())


        if self.src_item_object:
            self.src_item_json = self.src_item_object.to_json()

        # if self.check_redirect_list_url:
        try:
            if self.check_edit_permission:
                has_permission = self.check_edit_permission(self.src_item_object)
                if not has_permission:
                    self.update_redirect()
                    url = self.get_redirect()
                    return redirect(url)

        except Exception as e:
            print(e)
            flash(str(e), 'warning')
            self.update_redirect()
            return redirect(self.get_redirect())
            # return redirect(self.check_redirect_list_url)

        widgets = self._edit(pk)

        if not widgets:
            return self.post_edit_redirect()
        else:
            return self.render_template(
                self.edit_template,
                title=self.edit_title,
                widgets=widgets,
                related_views=self._related_views,
            )


    @event_logger.log_this
    @expose("/delete/<pk>")
    @has_access
    def delete(self, pk):
        pk = self._deserialize_pk_if_composite(pk)
        self.src_item_object = self.datamodel.get(pk, self._base_filters)
        if self.src_item_object:
            self.src_item_json = self.src_item_object.to_json()
        if self.check_redirect_list_url:
            try:
                if self.check_edit_permission:
                    if not self.check_edit_permission(self.src_item_object):
                        flash(str('no permission delete'), 'warning')
                        return redirect(self.check_redirect_list_url)
            except Exception as e:
                print(e)
                flash(str(e), 'warning')
                return redirect(self.check_redirect_list_url)
        self._delete(pk)
        url = url_for(f"{self.endpoint}.list")
        return redirect(url)
        # return self.post_delete_redirect()

from flask_appbuilder.widgets import GroupFormListWidget, ListMasterWidget
from flask import (
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    request,
    send_file,
    session,
    url_for,
)

class CompactCRUDMixin(BaseCRUDView):
    """
        Mix with ModelView to implement a list with add and edit on the same page.
    """

    @classmethod
    def set_key(cls, k, v):
        """Allows attaching stateless information to the class using the
        flask session dict
        """
        k = cls.__name__ + "__" + k
        session[k] = v

    @classmethod
    def get_key(cls, k, default=None):
        """Matching get method for ``set_key``
        """
        k = cls.__name__ + "__" + k
        if k in session:
            return session[k]
        else:
            return default

    @classmethod
    def del_key(cls, k):
        """Matching get method for ``set_key``
        """
        k = cls.__name__ + "__" + k
        session.pop(k)

    def _get_list_widget(self, **args):
        """ get joined base filter and current active filter for query """
        widgets = super(CompactCRUDMixin, self)._get_list_widget(**args)
        session_form_widget = self.get_key("session_form_widget", None)

        form_widget = None
        if session_form_widget == "add":
            form_widget = self._add().get("add")
        elif session_form_widget == "edit":
            pk = self.get_key("session_form_edit_pk")
            if pk and self.datamodel.get(int(pk)):
                form_widget = self._edit(int(pk)).get("edit")
        return {
            "list": GroupFormListWidget(
                list_widget=widgets.get("list"),
                form_widget=form_widget,
                form_action=self.get_key("session_form_action", ""),
                form_title=self.get_key("session_form_title", ""),
            )
        }

    @expose("/list/", methods=["GET", "POST"])
    @has_access
    def list(self):
        list_widgets = self._list()
        return self.render_template(
            self.list_template, title=self.list_title, widgets=list_widgets
        )

    @expose("/delete/<pk>")
    @has_access
    def delete(self, pk):
        pk = self._deserialize_pk_if_composite(pk)
        self._delete(pk)
        edit_pk = self.get_key("session_form_edit_pk")
        if pk == edit_pk:
            self.del_key("session_form_edit_pk")
        return redirect(self.get_redirect())


# 可以多选的列表页面
class ListWidgetWithCheckboxes(ListWidget):
    """An alternative to list view that renders Boolean fields as checkboxes

    Works in conjunction with the `checkbox` view."""

    template = "myapp/fab_overrides/list_with_checkboxes.html"


def validate_json(form, field):  # noqa
    try:
        json.loads(field.data)
    except Exception as e:
        logging.exception(e)
        raise Exception(_("json isn't valid"))


class YamlExportMixin(object):
    @action("yaml_export", __("Export to YAML"), __("Export to YAML?"), "fa-download")
    def yaml_export(self, items):
        if not isinstance(items, list):
            items = [items]

        data = [t.export_to_dict() for t in items]
        return Response(
            yaml.safe_dump(data),
            headers=generate_download_headers("yaml"),
            mimetype="application/text",
        )


# 列表页面删除/批量删除的操作
class DeleteMixin(object):
    def _delete(self, pk):
        """
            Delete function logic, override to implement diferent logic
            deletes the record with primary_key = pk

            :param pk:
                record primary key to delete
        """
        item = self.datamodel.get(pk, self._base_filters)
        if not item:
            abort(404)
        try:
            self.pre_delete(item)
        except Exception as e:
            flash(str(e), "danger")
        else:
            view_menu = security_manager.find_view_menu(item.get_perm())
            pvs = (
                security_manager.get_session.query(
                    security_manager.permissionview_model
                )
                .filter_by(view_menu=view_menu)
                .all()
            )

            schema_view_menu = None
            if hasattr(item, "schema_perm"):
                schema_view_menu = security_manager.find_view_menu(item.schema_perm)

                pvs.extend(
                    security_manager.get_session.query(
                        security_manager.permissionview_model
                    )
                    .filter_by(view_menu=schema_view_menu)
                    .all()
                )

            if self.datamodel.delete(item):
                self.post_delete(item)

                for pv in pvs:
                    security_manager.get_session.delete(pv)

                if view_menu:
                    security_manager.get_session.delete(view_menu)

                if schema_view_menu:
                    security_manager.get_session.delete(schema_view_menu)

                security_manager.get_session.commit()

            flash(*self.datamodel.message)
            self.update_redirect()

    @action(
        "muldelete", __("Delete"), __("Delete all Really?"), "fa-trash", single=False
    )
    def muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            try:
                self.pre_delete(item)
            except Exception as e:
                flash(str(e), "danger")
            else:
                self._delete(item.id)
        self.update_redirect()
        return redirect(self.get_redirect())


# model list的过滤器
class MyappFilter(BaseFilter):

    """Add utility function to make BaseFilter easy and fast

    These utility function exist in the SecurityManager, but would do
    a database round trip at every check. Here we cache the role objects
    to be able to make multiple checks but query the db only once
    """

    def get_user_roles(self):
        return get_user_roles()

    def get_all_permissions(self):
        """Returns a set of tuples with the perm name and view menu name"""
        perms = set()
        for role in self.get_user_roles():
            for perm_view in role.permissions:
                t = (perm_view.permission.name, perm_view.view_menu.name)
                perms.add(t)
        return perms

    def has_role(self, role_name_or_list):
        """Whether the user has this role name"""
        if not isinstance(role_name_or_list, list):
            role_name_or_list = [role_name_or_list]
        return any([r.name in role_name_or_list for r in self.get_user_roles()])

    def has_perm(self, permission_name, view_menu_name):
        """Whether the user has this perm"""
        return (permission_name, view_menu_name) in self.get_all_permissions()

    # 获取所有绑定了指定权限的所有vm
    def get_view_menus(self, permission_name):
        """Returns the details of view_menus for a perm name"""
        vm = set()
        for perm_name, vm_name in self.get_all_permissions():
            if perm_name == permission_name:
                vm.add(vm_name)
        return vm



# 检查是否有权限
def check_ownership(obj, raise_if_false=True):
    """Meant to be used in `pre_update` hooks on models to enforce ownership

    Admin have all access, and other users need to be referenced on either
    the created_by field that comes with the ``AuditMixin``, or in a field
    named ``owners`` which is expected to be a one-to-many with the User
    model. It is meant to be used in the ModelView's pre_update hook in
    which raising will abort the update.
    """
    if not obj:
        return False

    security_exception = MyappSecurityException(
        "You don't have the rights to alter [{}]".format(obj)
    )

    if g.user.is_anonymous:
        if raise_if_false:
            raise security_exception
        return False
    roles = [r.name for r in get_user_roles()]
    if "Admin" in roles:
        return True
    session = db.create_scoped_session()
    orig_obj = session.query(obj.__class__).filter_by(id=obj.id).first()

    # Making a list of owners that works across ORM models
    owners = []
    if hasattr(orig_obj, "owners"):
        owners += orig_obj.owners
    if hasattr(orig_obj, "owner"):
        owners += [orig_obj.owner]
    if hasattr(orig_obj, "created_by"):
        owners += [orig_obj.created_by]

    owner_names = [o.username for o in owners if o]

    if g.user and hasattr(g.user, "username") and g.user.username in owner_names:
        return True
    if raise_if_false:
        raise security_exception
    else:
        return False


# 绑定字段
def bind_field(
    self, form: DynamicForm, unbound_field: UnboundField, options: Dict[Any, Any]
) -> Field:
    """
    Customize how fields are bound by stripping all whitespace.

    :param form: The form
    :param unbound_field: The unbound field
    :param options: The field options
    :returns: The bound field
    """

    filters = unbound_field.kwargs.get("filters", [])
    filters.append(lambda x: x.strip() if isinstance(x, str) else x)
    return unbound_field.bind(form=form, filters=filters, **options)


FlaskForm.Meta.bind_field = bind_field
