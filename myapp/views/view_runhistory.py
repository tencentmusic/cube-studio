import datetime
import random
import json
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
import pysnooper
import urllib.parse
from flask import request, g
from flask_appbuilder.actions import action
from myapp.models.model_job import RunHistory, Pipeline
import calendar
from myapp import app, appbuilder, db
from wtforms import SelectField
from flask_appbuilder.fieldwidgets import Select2Widget
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from sqlalchemy import or_
from flask_appbuilder import expose
from .baseApi import (
    MyappModelRestApi
)
from flask import jsonify
from myapp import security_manager
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)

conf = app.config


class RunHistory_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query

        pipeline_ids = security_manager.get_create_pipeline_ids(db.session)
        return query.filter(
            or_(
                self.model.pipeline_id.in_(pipeline_ids),
                # self.model.project.name.in_(['public'])
            )
        )


class RunHistory_ModelView_Base():
    label_title = _('定时调度历史')
    datamodel = SQLAInterface(RunHistory)
    base_order = ('id', 'desc')
    order_columns = ['id']
    base_permissions = ['can_show', 'can_list', 'can_delete']
    list_columns = ['pipeline_url', 'creator', 'execution_date', 'status_url']
    cols_width = {
        "pipeline_url": {"type": "ellip2", "width": 300},
        "create_time": {"type": "ellip2", "width": 250}
    }
    edit_columns = ['status']
    base_filters = [["id", RunHistory_Filter, lambda: []]]
    add_form_extra_fields = {
        "status": SelectField(
            _('状态'),
            description= _("状态comed为已识别未提交，created为已提交"),
            widget=Select2Widget(),
            choices=[['comed', 'comed'], ['created', 'created']]
        ),
    }
    edit_form_extra_fields = add_form_extra_fields

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        return self._muldelete(items)


# 添加api
class RunHistory_ModelView_Api(RunHistory_ModelView_Base, MyappModelRestApi):
    datamodel = SQLAInterface(RunHistory)
    route_base = '/runhistory_modelview/api'


appbuilder.add_api(RunHistory_ModelView_Api)
