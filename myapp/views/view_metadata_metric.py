from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from sqlalchemy import or_
from flask_appbuilder.actions import action
from wtforms.validators import DataRequired, Regexp
from myapp import app, appbuilder
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget, MyBS3TextFieldWidget
import json
from .baseApi import (
    MyappModelRestApi
)
from flask import (
    abort,
    flash,
    g
)

from .base import (
    MyappFilter,
)
from myapp.models.model_metadata_metric import Metadata_metric

conf = app.config


class Metadata_Metrics_table_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query
        # 公开的，或者责任人包含自己的
        return query.filter(
            or_(
                self.model.public == True,
                self.model.metric_responsible.contains(g.user.username)
            )
        )


Remark_fields = {
    "begin_date": StringField(
        label= _("起点时间"),
        default='',
        description= _('起点时间精确到天'),
        widget=MyBS3TextFieldWidget(is_date=True)
    ),
    "end_date": StringField(
        label= _('终点时间'),
        default='',
        description= _(''),
        widget=MyBS3TextFieldWidget(is_date=True),
    ),
    "tip": StringField(
        label= _('备注'),
        description='',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    )
}


class Metadata_metric_ModelView_base():
    label_title = _('指标')
    datamodel = SQLAInterface(Metadata_metric)
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']
    base_order = ("id", "desc")
    order_columns = ['id']
    search_columns = ['metric_data_type', 'metric_responsible', 'app', 'name', 'label', 'describe', 'metric_type',
                      'metric_level', 'task_id', 'caliber', 'remark']
    show_columns = ['id', 'app', 'metric_data_type', 'name', 'label', 'describe', 'metric_type', 'metric_level',
                    'metric_dim', 'metric_responsible', 'caliber', 'task_id', 'public', 'remark']
    list_columns = ['app', 'metric_data_type', 'name', 'label', 'describe', 'metric_level', 'metric_responsible',
                    'public', 'metric_type', 'task_id']
    cols_width = {
        "name": {"type": "ellip2", "width": 200},
        "label": {"type": "ellip2", "width": 200},
        "task_id": {"type": "ellip2", "width": 250},
        "describe": {"type": "ellip2", "width": 400},
        "metric_responsible": {"type": "ellip2", "width": 300},
        "remark_html": {"type": "ellip1", "width": 300}
    }
    spec_label_columns = {
        "metric_data_type": _("指标模块"),
    }
    add_columns = ['app', 'metric_data_type', 'name', 'label', 'describe', 'metric_type', 'metric_level', 'metric_dim',
                   'metric_responsible', 'caliber', 'task_id', 'public', 'remark']
    # show_columns = ['project','name','describe','config_html','dag_json_html','expand_html']
    edit_columns = add_columns
    base_filters = [["id", Metadata_Metrics_table_Filter, lambda: []]]
    add_form_extra_fields = {
        "app": SelectField(
            label= _('产品'),
            description='',
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            choices=[[x, x] for x in ['app1', "app2", "app3"]]
        ),
        "name": StringField(
            label= _('名称'),
            description= _('指标英文名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(),Regexp('^[\x00-\x7F]*$')]
        ),
        "label": StringField(
            label= _('标签'),
            description= _('指标中文名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "describe": StringField(
            label= _("描述"),
            description= _('指标描述'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "remark": StringField(
            label= _('备注'),
            description= _('指标备注'),
            default='',
            widget=MyBS3TextAreaFieldWidget(expand_filed=Remark_fields)
        ),
        "metric_type": SelectField(
            label= _('指标类型'),
            description='',
            default= _('原子指标'),
            widget=Select2Widget(),
            choices=[[_(x), _(x)] for x in ['原子指标', '衍生指标']]
        ),
        "metric_data_type": SelectField(
            label= _('归属模块'),
            description= _('指标归属模块'),
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            choices=[[x, x] for x in ['module1', "module2", "module3"]]
        ),
        "metric_level": SelectField(
            label= _('指标等级'),
            description= _('指标重要级别'),
            default= _('普通'),
            widget=Select2Widget(),
            choices=[[_(x), _(x)] for x in ['普通', '重要', '核心']]
        ),
        "metric_dim": SelectField(
            label= _('指标维度'),
            description= _('指标维度'),
            default= 'day',
            widget=Select2Widget(),
            choices=[[x, x] for x in ['day', 'week', 'month']]
        ),
        "metric_responsible": StringField(
            label= _('指标责任人'),
            description= _('指标负责人，逗号分隔'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),
        "status": SelectField(
            label= _('状态'),
            description= _('指标状态'),
            widget=Select2Widget(),
            choices=[[_(x), _(x)] for x in ["下线", "待审批", '创建中', "上线", ]]
        ),
        "caliber": StringField(
            label= _('口径'),
            description= _('指标口径描述，代码和计算公式'),
            widget=MyBS3TextAreaFieldWidget(rows=3),
            validators=[DataRequired()]
        ),
        "task_id": StringField(
            label= _('任务id'),
            description='',
            widget=BS3TextFieldWidget()
        ),
    }

    edit_form_extra_fields = add_form_extra_fields
    import_data = True
    download_data = True

    def pre_show_res(self, _response):
        data = _response['data']
        if data.get('remark', None):
            data['remark'] = json.loads(data.get('remark', '[]'))

    def pre_add_req(self, req_json=None, *args, **kwargs):
        if req_json and 'remark' in req_json:
            req_json['remark'] = json.dumps(req_json.get('remark', []), indent=4, ensure_ascii=False)
        return req_json

    pre_update_req = pre_add_req

    add_fieldsets = [
        (
            _('指标'),
            {"fields": ['app','metric_data_type','name','label','describe','metric_type','metric_level','metric_dim','metric_responsible','caliber','task_id','public'], "expanded": True},
        ),
        (
            _('备注'),
            {"fields": ['remark'],
             "expanded": True},
        )
    ]
    edit_fieldsets = add_fieldsets

    def pre_upload(self, data):
        data['public'] = bool(int(data.get('public', 1)))
        return data

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            try:
                if g.user.is_admin() or item.created_by.username == g.user.username:
                    self.pre_delete(item)
                    self.datamodel.delete(item, raise_exception=True)
                    self.post_delete(item)
            except Exception as e:
                flash(str(e), "error")


class Metadata_metric_ModelView_Api(Metadata_metric_ModelView_base, MyappModelRestApi):
    datamodel = SQLAInterface(Metadata_metric)
    route_base = '/metadata_metric_modelview/api'


appbuilder.add_api(Metadata_metric_ModelView_Api)
