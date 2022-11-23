from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from wtforms.validators import DataRequired, Length, Regexp
from myapp import app, appbuilder
from myapp.utils import core
from wtforms import StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea, MySelectMultipleField


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
logging = app.logger





Metadata_column_fields = {
    "name":StringField(
        label=_("列名"),
        default='',
        description='列名(小写字母、数字、_ 组成)，最长50个字符',
        widget=BS3TextFieldWidget(),
        validators=[Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), Length(1, 54),DataRequired()]
    ),

    "describe": StringField(
        label=_('列描述'),
        default='',
        description='列名描述',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    ),
    "column_type": SelectField(
        label=_('字段类型'),
        description='列类型',
        widget=Select2Widget(),
        choices=[['int', 'int'], ['string', 'string'],['float', 'float']],
        validators=[DataRequired()]
    ),
    "remark": StringField(
        label=_('备注'),
        description='备注',
        default='',
        widget=BS3TextFieldWidget(),
    ),
    "partition_type": SelectField(
        label=_('列分区类型'),
        description='字段分区类型',
        widget=Select2Widget(),
        choices=[['主分区', '主分区'], ['子分区', '子分区'],['非分区', '非分区']],
        validators=[DataRequired()]
    ),
}



class Metadata_table_ModelView_base():
    label_title='元数据 表'
    datamodel = SQLAInterface(Metadata_table)
    base_permissions = ['can_add','can_show','can_edit','can_list','can_delete']

    base_order = ("id", "desc")
    order_columns = ['id','storage_cost','visits_seven']

    add_columns = ['app', 'db', 'table', 'describe', 'field', 'warehouse_level','value_score', 'storage_cost', 'security_level', 'ttl','create_table_ddl']
    show_columns = ['app','db','table','describe','field','warehouse_level','owner','c_org_fullname','storage_size','lifecycle','rec_lifecycle','storage_cost','visits_seven','recent_visit','partition_start','partition_end','status','visits_thirty','create_table_ddl','metadata_column']
    search_columns=['app','db','table','describe','field','warehouse_level','owner']
    spec_label_columns = {
        "table":"表名",
        "metadata_column":"列信息",
        "field": "数据域",
    }

    edit_columns = add_columns
    list_columns = ['app','db', 'table', 'owner','describe','field','warehouse_level', 'storage_cost']
    cols_width = {
        "app": {"type": "ellip2", "width": 150},
        "db":{"type": "ellip2", "width": 250},
        "table":{"type": "ellip2", "width": 400},
        "owner":{"type": "ellip2", "width": 150},
        "field": {"type": "ellip2", "width": 150},
        "describe": {"type": "ellip2", "width": 300},
        "warehouse_level": {"type": "ellip2", "width": 150},
        "storage_cost":{"type": "ellip2", "width": 200},
        "visits_seven":{"type": "ellip2", "width": 200},
        "visits_thirty":{"type": "ellip2", "width": 200}
    }

    add_form_extra_fields = {
        "table":StringField(
            label=_('表名'),
            default='',
            description='数据表 格式：dwd_[产品]_[数据域]_[数据域描述]_[刷新周期d/w/m/y][存储策略i(增量)/和f(全量)]  例如，dwd_qq_common_click_di; 表名由字母数组下划线组成 ',
            widget=BS3TextFieldWidget(),
            validators=[Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), Length(1, 54),DataRequired()]
        ),
        "describe": StringField(
            label=_(datamodel.obj.lab('describe')),
            description='表格描述',
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

        "app": SelectField(
            label=_(datamodel.obj.lab('app')),
            description='产品分类',
            widget=MySelect2Widget(can_input=True,conten2choices=True),
            default='',
            choices=[[x,x] for x in ['产品1',"产品2","产品3"]],
            validators=[DataRequired()]
        ),
        "db": SelectField(
            label=_(datamodel.obj.lab('db')),
            description='数据库名称',
            widget=MySelect2Widget(can_input=True,conten2choices=True),
            choices=[[]],
            validators=[DataRequired()]
        ),
        "field":MySelectMultipleField(
            label=_(datamodel.obj.lab('field')),
            description='数据域',
            widget=MySelect2Widget(can_input=True,conten2choices=True),
            choices=[[x,x] for x in ['数据域1',"数据域2","数据域3"]],
            validators=[]
        ),
        "security_level": SelectField(
            label=_(datamodel.obj.lab('security_level')),
            description='安全等级',
            widget=Select2Widget(),
            default='普通',
            choices=[[x,x] for x in ["普通", "机密","秘密","高度机密"]],
            validators=[DataRequired()]
        ),
        "value_score": StringField(
            label=_(datamodel.obj.lab('value_score')),
            description='价值评分',
            widget=BS3TextFieldWidget(),
        ),
        "storage_size": StringField(
            label=_(datamodel.obj.lab('storage_size')),
            description='存储大小',
            widget=BS3TextFieldWidget(),
        ),
        "warehouse_level": SelectField(
            label=_(datamodel.obj.lab('warehouse_level')),
            default='TMP',
            description='数仓等级',
            widget=Select2Widget(),
            choices=[["ODS",'ODS'],["DWD",'DWD'],["DWS",'DWS'],["TOPIC",'TOPIC'],['APP','APP'],["DIM",'DIM'],["TMP",'TMP']],
            validators=[DataRequired()]
        ),
        "storage_cost": StringField(
            label=_(datamodel.obj.lab('cost')),
            description='数据成本',
            widget=BS3TextFieldWidget(),
        ),
        "owner": StringField(
            label=_(datamodel.obj.lab('owner')),
            default='',
            description='责任人,逗号分隔的多个用户',
            widget=BS3TextFieldWidget(),
        ),
        "ttl": SelectField(
            label=_(datamodel.obj.lab('ttl')),
            description='保留周期',
            widget=Select2Widget(),
            default='一年',
            choices=[[x,x] for x in ["一周", "一个月","三个月","半年","一年","永久"]],
            validators=[DataRequired()]
        ),
        "sql_demo": StringField(
            _(datamodel.obj.lab('sql_demo')),
            description='建表sql 示例',
            widget=MyCodeArea(code=core.hive_create_sql_demo()),  # 传给widget函数的是外层的field对象，以及widget函数的参数
        ),
        "create_table_ddl": StringField(
            label='建表sql',
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
            description='建表sql语句',
            widget=MyBS3TextAreaFieldWidget(rows=10)
        )
    }

    edit_form_extra_fields = add_form_extra_fields


    @action(
        "muldelete", __("Delete"), __("Delete all Really?"), "fa-trash", single=False
    )
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
        item.node_id=item.db+"::"+item.table
        item.creator = g.user.username


    # @event_logger.log_this
    @action("ddl", __("更新到远程hive表"), __("ddl 保存修改"), "fa-save", multiple=False, single=True)
    def ddl(self, item):
        pass
        # 自己实现更新到hive表


class Metadata_table_ModelView_Api(Metadata_table_ModelView_base,MyappModelRestApi,DeleteMixin):
    datamodel = SQLAInterface(Metadata_table)
    route_base = '/metadata_table_modelview/api'



    # @pysnooper.snoop()
    def pre_add_web(self):
        self.default_filter = {
            "owner": g.user.username
        }

    # @pysnooper.snoop()
    def pre_list_res(self,result):
        data = result['data']
        for item in data:
            storage_cost = item.get('storage_cost',0)
            if storage_cost:
                item['storage_cost']=round(float(storage_cost), 6)
        return result

    # # 在info信息中添加特定参数
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        pass

    remember_columns=['app','db']
    label_title='hive库表'

appbuilder.add_api(Metadata_table_ModelView_Api)

