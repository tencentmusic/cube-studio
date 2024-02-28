from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from wtforms.validators import DataRequired, Length, Regexp
from myapp import app, appbuilder
from myapp.utils import core
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget, MySelect2Widget, MyCodeArea, MySelectMultipleField

from .baseApi import (
    MyappModelRestApi,
)
from flask import (
    abort,
    flash,
    g
)
from .base import (
    DeleteMixin
)

from myapp.models.model_metadata import Metadata_table

conf = app.config

Metadata_column_fields = {
    "name": StringField(
        label= _("列名"),
        default='',
        description= _('列名(小写字母、数字、_ 组成)，最长50个字符'),
        widget=BS3TextFieldWidget(),
        validators=[Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), Length(1, 54), DataRequired()]
    ),

    "describe": StringField(
        label= _('列描述'),
        default='',
        description= _('列名描述'),
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    ),
    "column_type": SelectField(
        label= _('字段类型'),
        description= _('列类型'),
        widget=Select2Widget(),
        choices=[['int', 'int'], ['string', 'string'], ['float', 'float']],
        validators=[DataRequired()]
    ),
    "remark": StringField(
        label= _('备注'),
        description= _('备注'),
        default='',
        widget=BS3TextFieldWidget(),
    ),
    "partition_type": SelectField(
        label= _('列分区类型'),
        description= _('字段分区类型'),
        widget=Select2Widget(),
        choices=[[_(x), _(x)] for x in ['主分区', "子分区", "非分区"]],
        validators=[DataRequired()]
    ),
}


class Metadata_table_ModelView_base():
    label_title = _('元数据 表')
    datamodel = SQLAInterface(Metadata_table)
    base_permissions = ['can_add', 'can_show', 'can_edit', 'can_list', 'can_delete']

    base_order = ("id", "desc")
    order_columns = ['id', 'storage_cost', 'visits_seven']

    add_columns = ['app', 'db', 'table', 'describe', 'field', 'warehouse_level', 'value_score', 'storage_cost',
                   'security_level', 'ttl', 'create_table_ddl']
    show_columns = ['app', 'db', 'table', 'describe', 'field', 'warehouse_level', 'owner', 'c_org_fullname',
                    'storage_size', 'lifecycle', 'rec_lifecycle', 'storage_cost', 'visits_seven', 'recent_visit',
                    'partition_start', 'partition_end', 'status', 'visits_thirty', 'create_table_ddl',
                    'metadata_column']
    search_columns = ['app', 'db', 'table', 'describe', 'field', 'warehouse_level', 'owner']
    spec_label_columns = {
        "field": _("数据域"),
    }

    edit_columns = add_columns
    list_columns = ['app', 'db', 'table', 'owner', 'describe', 'field', 'warehouse_level', 'storage_cost']
    cols_width = {
        "app": {"type": "ellip2", "width": 150},
        "db": {"type": "ellip2", "width": 200},
        "table": {"type": "ellip2", "width": 300},
        "owner": {"type": "ellip2", "width": 150},
        "field": {"type": "ellip2", "width": 150},
        "describe": {"type": "ellip2", "width": 300},
        "warehouse_level": {"type": "ellip2", "width": 150},
        "storage_cost": {"type": "ellip2", "width": 200},
        "visits_seven": {"type": "ellip2", "width": 200},
        "visits_thirty": {"type": "ellip2", "width": 200}
    }

    import_data = True
    download_data = True

    def pre_upload(self,data):

        if not data.get('recent_visit',None):
            data['recent_visit']=None
        return data

    add_form_extra_fields = {
        "table": StringField(
            label= _('表名'),
            default='',
            description= _('数据表 格式：dwd_[产品]_[数据域]_[数据域描述]_[刷新周期d/w/m/y][存储策略i(增量)/和f(全量)]  例如，dwd_qq_common_click_di; 表名由字母数组下划线组成 '),
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), Length(1, 54), DataRequired()]
        ),
        "describe": StringField(
            label= _("描述"),
            description='',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "app": SelectField(
            label= _('产品'),
            description='',
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            default='',
            choices=[[x, x] for x in ['app1', "app2", "app3"]],
            validators=[DataRequired()]
        ),
        "db": SelectField(
            label= _('数据库'),
            description='',
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            choices=[[]],
            validators=[DataRequired()]
        ),
        "field": MySelectMultipleField(
            label= _('数据域'),
            description='',
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            choices=[[x, x] for x in ['field1', "field2", "field3"]],
            validators=[]
        ),
        "security_level": SelectField(
            label= _('安全等级'),
            description='',
            widget=Select2Widget(),
            default= _('普通'),
            choices=[[_(x), _(x)] for x in ["普通", "机密", "秘密", "高度机密"]],
            validators=[DataRequired()]
        ),
        "value_score": StringField(
            label= _('价值评分'),
            description='',
            widget=BS3TextFieldWidget(),
        ),
        "storage_size": StringField(
            label= _('存储大小'),
            description='',
            widget=BS3TextFieldWidget(),
        ),
        "warehouse_level": SelectField(
            label= _('数仓等级'),
            default='TMP',
            description= _('数仓等级'),
            widget=Select2Widget(),
            choices=[["ODS",'ODS'],["DWD",'DWD'],["DWS",'DWS'],["TOPIC",'TOPIC'],['APP','APP'],["DIM",'DIM'],["TMP",'TMP']],
            validators=[DataRequired()]
        ),
        "storage_cost": StringField(
            label= _('数据成本'),
            description='',
            widget=BS3TextFieldWidget(),
        ),
        "owner": StringField(
            label= _('责任人'),
            default='',
            description= _('责任人,逗号分隔的多个用户'),
            widget=BS3TextFieldWidget(),
        ),
        "ttl": SelectField(
            label= _('保留周期'),
            description='',
            widget=Select2Widget(),
            default= '1 year',
            choices=[[_(x), _(x)] for x in ["1 week", "1 month", "3 month", "6 month", "1 year", "forever"]],
            validators=[DataRequired()]
        ),
        "sql_demo": StringField(
            _('sql 示例'),
            description= _('建表sql 示例'),
            widget=MyCodeArea(code=core.hive_create_sql_demo()),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "create_table_ddl": StringField(
            label= _('建表sql'),
            #             default='''
            # -- 建表示例sql
            # use {db_name};
            # CREATE TABLE if not exists {table_name}(
            # imp_date int COMMENT '统计日期',
            # ori_log string COMMENT '原始日志',
            # fint int COMMENT '某个数字字段'
            # )
            # PARTITION BY LIST(imp_date)
            # (PARTITION default)
            # STORED AS ORCFILE COMPRESS;
            # '''
            description= _('建表sql语句'),
            widget=MyBS3TextAreaFieldWidget(rows=10)
        )
    }

    edit_form_extra_fields = add_form_extra_fields

    def check_edit_permission(self,item):
        if not g.user.is_admin() and g.user.username not in item.owner:
            return False
        return True
    check_delete_permission = check_edit_permission

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        if not items:
            abort(404)
        for item in items:
            try:
                if g.user.is_admin() or (item.owner and g.user.username in item.owner):
                    self.datamodel.delete(item, raise_exception=True)
            except Exception as e:
                flash(str(e), "danger")

    # @pysnooper.snoop(watch_explode=('item'))
    def pre_add(self, item):
        # 建表
        item.owner = g.user.username
        item.node_id = item.db + "::" + item.table
        item.creator = g.user.username


    # # @event_logger.log_this
    # @action("ddl", "更新远程表", "如果更新失败，请手动更改远程数据库的表结构", "fa-save", multiple=False, single=True)
    # def ddl(self, item):
    #     pass
    #     # 自己实现更新到hive表



class Metadata_table_ModelView_Api(Metadata_table_ModelView_base, MyappModelRestApi, DeleteMixin):
    datamodel = SQLAInterface(Metadata_table)
    route_base = '/metadata_table_modelview/api'

    # @pysnooper.snoop()
    def pre_add_web(self):
        self.default_filter = {
            "owner": g.user.username
        }

    # @pysnooper.snoop()
    def pre_list_res(self, result):
        data = result['data']
        for item in data:
            storage_cost = item.get('storage_cost', 0)
            if storage_cost:
                item['storage_cost'] = round(float(storage_cost), 6)
        return result

    # # 在info信息中添加特定参数
    # @pysnooper.snoop()
    def add_more_info(self, response, **kwargs):
        pass

    remember_columns = ['app', 'db']
    label_title = _('库表')


appbuilder.add_api(Metadata_table_ModelView_Api)
