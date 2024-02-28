import random
import time
import logging
import pandas
from myapp.views.baseSQLA import MyappSQLAInterface as SQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
import copy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from flask_appbuilder.actions import action
import os
from flask import jsonify, make_response
from sqlalchemy import or_
from wtforms.validators import DataRequired, Length, Regexp
from myapp import app, appbuilder, db
from wtforms import BooleanField, StringField, SelectField
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, Select2Widget
from myapp.forms import MySelect2Widget, MyBS3TextAreaFieldWidget
from .baseApi import MyappModelRestApi
from flask import (
    abort,
    flash,
    g,
    Markup,
    redirect,
    request
)
from .base import MyappFilter
from myapp.models.model_dimension import Dimension_table
from flask_appbuilder import expose
import pysnooper, datetime, json

conf = app.config


class Dimension_table_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        if g.user.is_admin():
            return query.filter(self.model.status == 1)

        return query.filter(self.model.status == 1).filter(
            or_(
                self.model.owner.contains(g.user.username),
                self.model.owner.contains('*'),
            )
        )


Metadata_column_fields = {
    "name": StringField(
        label= _("列名"),
        description= _('列名(小写字母、数字、_ 组成)，最长50个字符'),
        default='',
        widget=BS3TextFieldWidget(),
        validators=[Regexp("^[a-z][a-z0-9_]*[a-z0-9]$"), Length(1, 54), DataRequired()]
    ),
    "describe": StringField(
        label= _('列描述'),
        description='',
        default='',
        widget=BS3TextFieldWidget(),
        validators=[DataRequired()]
    ),
    "column_type": SelectField(
        label= _('字段类型'),
        description='',
        widget=Select2Widget(),
        default='text',
        choices=[['int', 'int'], ['text', 'text'], ['date', 'date'], ['double', 'double'], ['enum', 'enum']],
        validators=[DataRequired()]
    ),
    "unique": BooleanField(
        label= _('是否唯一'),
        description='',
        default=False,
        widget=BS3TextFieldWidget(),
    ),
    "nullable": BooleanField(
        label= _('是否可为空'),
        description='',
        default=True,
        widget=BS3TextFieldWidget(),
    ),
    "primary_key": BooleanField(
        label= _('是否为主键'),
        description='',
        default=False,
        widget=BS3TextFieldWidget(),
    ),
    "choices": StringField(
        label= _('可选择项'),
        description= _('enum类型时，逗号分割多个可选择项，为空则为数据库记录已存在可选择项'),
        default='',
        widget=BS3TextFieldWidget(),
    )
}


# @pysnooper.snoop()
def ddl_hive_external_table(table_id):
    try:
        item = db.session.query(Dimension_table).filter_by(id=int(table_id)).first()
        if not item:
            return
        cols = json.loads(item.columns)
        # 创建hive外表
        hive_type_map = {'INT': 'BIGINT', 'TEXT': 'STRING','STRING': 'STRING','DATE': 'STRING', 'DOUBLE': 'DOUBLE','ENUM':'STRING'}
        cols_lst = []
        for col_name in cols:
            if col_name in ['id', ]:
                continue
            column_type = cols[col_name].get('column_type', 'text').upper()
            if column_type not in hive_type_map:
                raise RuntimeError(__("更新了不支持的新字段类型"))
            column_type = hive_type_map[column_type]
            col_str = col_name + ' ' + column_type
            cols_lst.append(col_str)

        columns_sql = ',\n'.join(cols_lst).strip(',')
        import sqlalchemy.engine.url as url
        uri = url.make_url(item.sqllchemy_uri if item.sqllchemy_uri else default_uri)
        hive_sql = ''' 
# use your hive db;
CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}  (
id BIGINT,
{columns_sql}
) 
with (ip='{ip}',port='{port}',db_name='{pg_db_name}',user_name='{user_name}',pwd='{password}',table_name='{pg_table_name}',charset='utf8',db_type='pg');

                        '''.format(
            table_name=item.table_name,
            columns_sql=columns_sql,
            ip=uri.host,
            port=str(uri.port),
            user_name=uri.username,
            password=uri.password,
            pg_db_name=uri.database,
            pg_table_name=item.table_name
        )
        return hive_sql
        # 执行创建数据库的sql
        logging.info(hive_sql)
    except Exception as e:
        print(e)
        return str(e)


default_uri = 'mysql+pymysql://your_username:your_password@your_host:port/your_db'


class Dimension_table_ModelView_Api(MyappModelRestApi):
    datamodel = SQLAInterface(Dimension_table)
    label_title = _('维表')
    route_base = '/dimension_table_modelview/api'
    base_permissions = ['can_add', 'can_list', 'can_delete', 'can_show', 'can_edit']
    add_columns = ['sqllchemy_uri', 'app', 'table_name', 'label', 'describe', 'owner', 'columns']
    edit_columns = add_columns
    show_columns = ['id', 'app', 'sqllchemy_uri', 'label', 'describe', 'table_name', 'owner', 'columns', 'status']
    search_columns = ['id', 'app', 'table_name', 'label', 'describe', 'sqllchemy_uri']
    order_columns = ['id']
    base_order = ('id', 'desc')
    list_columns = ['table_html', 'label', 'owner', 'describe', 'operate_html']
    cols_width = {
        "table_html": {"type": "ellip2", "width": 300},
        "label": {"type": "ellip2", "width": 300},
        "owner": {"type": "ellip2", "width": 300},
        "describe": {"type": "ellip2", "width": 300},
        "operate_html": {"type": "ellip2", "width": 400}
    }
    spec_label_columns = {
        "sqllchemy_uri": _("链接串地址")
    }
    base_filters = [["id", Dimension_table_Filter, lambda: []]]

    add_fieldsets = [
        (
            _('表元数据'),
            {"fields": ['sqllchemy_uri', 'app', 'table_name', 'label', 'describe', 'owner'], "expanded": True},
        ),
        (
            _('列信息'),
            {"fields": ['columns'],
             "expanded": True},
        )
    ]
    edit_fieldsets = add_fieldsets

    add_form_extra_fields = {
        "sqllchemy_uri": StringField(
            _('链接串地址'),
            default="",
            description= _('链接串地址： <br> 示例：mysql+pymysql://$账号:$密码@$ip:$端口/$库名?charset=utf8 <br> 示例：postgresql+psycopg2://$账号:$密码@$ip:$端口/$库名'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired(), Regexp("^(mysql\+pymysql|postgresql\+psycopg2)://.*:.*@.*:[0-9]*/[a-zA-Z_\-]*")]
        ),
        "table_name": StringField(
            label= _('表名'),
            description= _('远程数据库的表名'),
            widget=BS3TextFieldWidget(),
            default='',
            validators=[DataRequired(), Regexp("^[a-z][a-z0-9_\-]*[a-z0-9]$")]
        ),
        "label": StringField(
            label= _('标签'),
            description='',
            widget=BS3TextFieldWidget(),
            default='',
            validators=[DataRequired()]
        ),
        "describe": StringField(
            label= _('描述'),
            description='',
            widget=BS3TextFieldWidget(),
            default='',
            validators=[DataRequired()]
        ),
        "app": SelectField(
            label= _('产品'),
            description='',
            widget=MySelect2Widget(can_input=True, conten2choices=True),
            default='',
            choices=[[_(x), _(x)] for x in ['app1', "app2", "app3"]],
            validators=[DataRequired()]
        ),
        "columns": StringField(
            label= _('字段信息'),
            description= _('维表字段信息，必须包含自增主键列，例如id'),
            widget=MyBS3TextAreaFieldWidget(expand_filed=Metadata_column_fields)
        ),
        "owner": StringField(
            label= _('责任人'),
            default='',
            description= _('责任人,逗号分隔的多个用户'),
            widget=BS3TextFieldWidget(),
            validators=[DataRequired()]
        ),

    }
    edit_form_extra_fields = add_form_extra_fields

    def pre_add(self, item):
        if not item.columns:
            item.columns = '{}'
        sqllchemy_uri = item.sqllchemy_uri if item.sqllchemy_uri else default_uri
        if item.columns:
            # 如果没有主键列就自动加上row主键列
            cols = json.loads(item.columns)
            for col_name in cols:
                if cols[col_name].get('primary_key', False):
                    return
            if 'postgresql' in sqllchemy_uri:
                cols['id'] = {
                    "column_type": "int",
                    "describe": __("主键"),
                    "name": "id",
                    "nullable": False,
                    "primary_key": True,
                    "unique": True
                }
            if 'mysql' in sqllchemy_uri:
                cols['id'] = {
                    "column_type": "int",
                    "describe": __("主键"),
                    "name": "id",
                    "nullable": False,
                    "primary_key": True,
                    "unique": True
                }
            # 不允许有自定义主键
            if not sqllchemy_uri or sqllchemy_uri == default_uri:
                for col_name in cols:
                    if cols[col_name].get("primary_key", False) and col_name != 'id':
                        cols[col_name]["primary_key"] = False

            item.columns = json.dumps(cols, indent=4, ensure_ascii=False)
        if not item.owner or g.user.username not in item.owner:
            item.owner = g.user.username if not item.owner else item.owner + "," + g.user.username

        flash(__('添加或修改字段类型，需要点击"更新远程表"，以实现在远程数据库上建表'), 'warning')

    def pre_update(self, item):
        if not item.sqllchemy_uri:
            item.sqllchemy_uri = self.src_item_json.get('sqllchemy_uri', '')
        self.pre_add(item)

        # 更新以后表结构会变
        all_dimension = conf.get('all_dimension_instance', {})
        if "dimension_%s" % item.id in all_dimension:
            del all_dimension["dimension_%s" % item.id]

    # 转换为前端list
    def pre_show_res(self, _response):
        data = _response['data']
        columns = json.loads(data.get('columns', '{}'))
        columns_list = []
        for name in columns:
            col = columns[name]
            col.update({"name": name})
            columns_list.append(col)
        data['columns'] = columns_list

    # 添加或者更新前将前端columns list转化为字段存储
    def pre_add_req(self, req_json=None):
        if req_json and 'columns' in req_json:
            columns = {}
            for col in req_json.get('columns', []):
                columns[col['name']] = col
            req_json['columns'] = json.dumps(columns, indent=4, ensure_ascii=False)
        return req_json

    pre_update_req = pre_add_req

    # 获取指定维表里面的数据
    @staticmethod
    def get_dim_target_data(dim_id):

        dim = db.session.query(Dimension_table).filter_by(id=int(dim_id)).first()
        import sqlalchemy.engine.url as url
        uri = url.make_url(dim.sqllchemy_uri if dim.sqllchemy_uri else default_uri)
        sql_engine = create_engine(uri)
        sql = 'select * from %s' % (dim.table_name,)
        results = pandas.read_sql_query(sql, sql_engine)
        return results.to_dict()

    @expose("/external/<dim_id>", methods=["GET"])
    def external(self, dim_id):
        ddl_sql = ddl_hive_external_table(dim_id)
        print(ddl_sql)
        return Markup(ddl_sql.replace('\n', '<br>'))

    # @expose("/clear/<dim_id>", methods=["GET"])
    @action("clear", "清空", "清空选中维表的所有远程数据?", "fa-trash", single=True, multiple=False)
    def delete_all(self, items):
        if not items:
            abort(404)
        dim_id = ''
        try:
            for dim in items:
                dim_id = dim.id
                import sqlalchemy.engine.url as url
                uri = url.make_url(dim.sqllchemy_uri if dim.sqllchemy_uri else default_uri)
                engine = create_engine(uri)
                dbsession = scoped_session(sessionmaker(bind=engine))
                dbsession.execute('TRUNCATE TABLE  %s;' % dim.table_name)
                dbsession.commit()
                dbsession.close()
                flash(__('清空完成'), 'success')
        except Exception as e:
            flash(__('清空失败：') + str(e), 'error')

        url_path = conf.get('MODEL_URLS', {}).get("dimension") + '?targetId=' + dim_id
        return redirect(url_path)

    @expose("/create_external_table/<dim_id>", methods=["GET"])
    # @pysnooper.snoop()
    def create_external_table(self, dim_id):
        item = db.session.query(Dimension_table).filter_by(id=int(dim_id)).first()
        sqllchemy_uri = item.sqllchemy_uri if item.sqllchemy_uri else default_uri
        if sqllchemy_uri:
            # 创建数据库的sql(如果数据库存在就不创建，防止异常)
            if 'postgresql' in sqllchemy_uri:

                # 创建pg表
                import sqlalchemy.engine.url as url
                uri = url.make_url(sqllchemy_uri)
                from sqlalchemy import create_engine
                from sqlalchemy.orm import scoped_session, sessionmaker
                engine = create_engine(uri)
                dbsession = scoped_session(sessionmaker(bind=engine))
                cols = json.loads(item.columns)
                table_schema = 'public'

                read_col_sql = r"select column_name from information_schema.columns where table_schema='%s' and table_name='%s' "%(table_schema,item.table_name)
                print(read_col_sql)
                company_data = pandas.read_sql(read_col_sql, con=engine)
                # 如果表不存在
                sql = ''
                if company_data.empty:
                    # 如果远程没有表，就建表
                    sql = '''
                    CREATE TABLE if not exists  {table_name}  (
                        id BIGINT PRIMARY KEY,
                        {columns_sql}
                    );
                                    '''.format(
                        table_name=item.table_name,
                        columns_sql='\n'.join(
                            ["    %s %s %s %s," % (col_name, 'BIGINT' if cols[col_name].get('column_type','text').upper() == 'INT' else 'double precision' if cols[col_name].get('column_type','text').upper() in ['DOUBLE','FLOAT'] else 'varchar(2000)',
                                                   '' if int(cols[col_name].get('nullable', True)) else 'NOT NULL',
                                                   '' if not int(cols[col_name].get('unique', False)) else 'UNIQUE') for
                             col_name in cols if col_name not in ['id', ]]
                        ).strip(',')
                    )
                    # 执行创建数据库的sql
                    print(sql)
                    if sql:
                        dbsession.execute(sql)
                        dbsession.commit()
                    flash(__('创建新表成功'), 'success')
                else:
                    exist_columns = list(company_data.head().to_dict()['column_name'].values())
                    print(exist_columns)
                    if exist_columns:
                        col = json.loads(item.columns)
                        for column_name in col:
                            col_type = 'INT' if col[column_name].get('column_type','text').upper() == 'INT' else 'varchar(2000)'
                            if column_name not in exist_columns:
                                try:
                                    sql = 'ALTER TABLE %s ADD %s %s;' % (item.table_name, column_name, col_type)
                                    print(sql)
                                    dbsession.execute(sql)
                                    dbsession.commit()
                                    flash(__('增加新字段成功'), 'success')
                                except Exception as e:
                                    dbsession.rollback()
                                    print(e)
                                    flash(__('增加新字段失败：') + str(e), 'error')

                dbsession.close()
                # 如果远程有表，就增加字段

            # 创建数据库的sql(如果数据库存在就不创建，防止异常)
            if 'mysql' in sqllchemy_uri:
                # 创建mysql表
                import sqlalchemy.engine.url as url
                uri = url.make_url(sqllchemy_uri)
                from sqlalchemy import create_engine
                from sqlalchemy.orm import scoped_session, sessionmaker
                engine = create_engine(uri)
                dbsession = scoped_session(sessionmaker(bind=engine))
                cols = json.loads(item.columns)

                import sqlalchemy
                try:
                    table = sqlalchemy.Table(item.table_name, sqlalchemy.MetaData(), autoload=True, autoload_with=engine)

                    exist_columns = [str(col).replace(item.table_name, '').replace('.', '') for col in table.c]
                    print(exist_columns)
                    if exist_columns:
                        col = json.loads(item.columns)
                        for column_name in col:
                            col_type = 'varchar(2000)'
                            if col[column_name].get('column_type', 'text').upper() == 'INT':
                                col_type = 'INT'
                            if col[column_name].get('column_type', 'text').upper() in ['DOUBLE', 'FLOAT']:
                                col_type = 'DOUBLE'
                            if column_name not in exist_columns:
                                try:
                                    sql = 'ALTER TABLE %s ADD %s %s;' % (item.table_name, column_name, col_type)
                                    print(sql)
                                    dbsession.execute(sql)
                                    dbsession.commit()
                                    flash(__('增加新字段成功'), 'success')
                                except Exception as e:
                                    dbsession.rollback()
                                    print(e)
                                    flash(__('增加新字段失败：') + str(e), 'error')


                except sqlalchemy.exc.NoSuchTableError:
                    print('表不存在')
                    # 如果表不存在

                    # 如果远程没有表，就建表
                    sql = '''
                    CREATE TABLE if not exists  {table_name}  (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        {columns_sql}
                    );
                                    '''.format(
                        table_name=item.table_name,
                        columns_sql='\n'.join(
                            ["    %s %s %s %s," % (col_name, 'BIGINT' if cols[col_name].get('column_type','text').upper() == 'INT' else 'DOUBLE' if cols[col_name].get('column_type','text').upper() in ['DOUBLE','FLOAT'] else 'varchar(2000)',
                                                   '' if int(cols[col_name].get('nullable', True)) else 'NOT NULL',
                                                   '' if not int(cols[col_name].get('unique', False)) else 'UNIQUE') for
                             col_name in cols if col_name not in ['id', ]]
                        ).strip(',')
                    )
                    # 执行创建数据库的sql
                    print(sql)
                    if sql:
                        dbsession.execute(sql)
                        dbsession.commit()
                        flash(__('创建新表成功'), 'success')

                dbsession.close()
                # 如果远程有表，就增加字段

        all_dimension = conf.get('all_dimension_instance', {})
        if "dimension_%s" % dim_id in all_dimension:
            del all_dimension["dimension_%s" % dim_id]

        url_path = conf.get('MODEL_URLS', {}).get("dimension") + '?targetId=' + dim_id
        return redirect(url_path)

    def add_more_info(self, response, **kwargs):
        from myapp.views.baseApi import API_RELATED_RIS_KEY, API_ADD_COLUMNS_RES_KEY, API_EDIT_COLUMNS_RES_KEY
        for col in response[API_ADD_COLUMNS_RES_KEY]:
            if col['name'] == 'columns':
                response[API_EDIT_COLUMNS_RES_KEY].remove(col)
        for col in response[API_EDIT_COLUMNS_RES_KEY]:
            if col['name'] == 'columns':
                response[API_EDIT_COLUMNS_RES_KEY].remove(col)

        response[API_ADD_COLUMNS_RES_KEY].append({
            "name": "columns",
            "ui-type": "list",
            "info": self.columnsfield2info(Metadata_column_fields)
        })
        response[API_EDIT_COLUMNS_RES_KEY].append({
            "name": 'columns',
            "ui-type": "list",
            "info": self.columnsfield2info(Metadata_column_fields)
        })


appbuilder.add_api(Dimension_table_ModelView_Api)

from flask_appbuilder import Model
from myapp.models.base import MyappModelBase
from sqlalchemy import Column, Integer, String, ForeignKey, Float, BigInteger, Date
from sqlalchemy.orm import relationship


class Dimension_remote_table_ModelView_Api(MyappModelRestApi):
    datamodel = SQLAInterface(Dimension_table)

    route_base = '/dimension_remote_table_modelview'

    # @pysnooper.snoop()
    def set_model(self, dim_id):

        dim = db.session.query(Dimension_table).filter_by(id=int(dim_id)).first()
        if not dim:
            raise Exception("no dimension")

        all_dimension = conf.get('all_dimension_instance', {})
        if 1 or "dimension_%s" % dim_id not in all_dimension:
            columns = json.loads(dim.columns) if dim.columns else {}
            column_class = {}

            spec_label_columns = {}
            search_columns = []
            label_columns = {}
            add_columns = []
            edit_columns = []
            show_columns = []
            list_columns = []
            description_columns = {}
            add_form_extra_fields = {}
            add_form_query_rel_fields = {}
            validators_columns = {}
            order_columns = []
            cols_width = {
            }
            for column_name in columns:
                column_type = columns[column_name].get('column_type', 'string')
                cols_width[column_type] = {
                    "type": "ellip1",
                    "width": max(100, len(columns[column_name].get("label", '')) * 20)
                }
                if column_type == 'int' or column_type == 'double':
                    cols_width[column_type] = {
                        "type": "ellip1",
                        "width": max(100,len(columns[column_name].get("label",''))*20)
                    }
                if column_type == 'date':
                    cols_width[column_type] = {
                        "type": "ellip1",
                        "width": max(200,len(columns[column_name].get("label",''))*20)
                    }
                if column_type == 'text':
                    cols_width[column_type] = {
                        "type": "ellip2",
                        "width": max(300,len(columns[column_name].get("label",''))*20)
                    }
                if column_type == 'enum':
                    cols_width[column_type] = {
                        "type": "ellip2",
                        "width": max(200,len(columns[column_name].get("label",''))*20)
                    }

                column_sql_type = BigInteger if column_type == 'int' else String  # 因为实际使用的时候，会在数据库中存储浮点数据，通用性也更强

                val = [DataRequired()] if not columns[column_name].get('nullable', True) else []
                if column_type == 'date':
                    column_sql_type = String
                    add_form_extra_fields[column_name] = StringField(
                        _(column_name),
                        default=datetime.datetime.now().strftime('%Y-%m-%d'),
                        description='',  # columns[column_name]['describe'],
                        widget=BS3TextFieldWidget(),
                        validators=[Regexp("^[0-9]{4,4}-[0-9]{2,4}-[0-9]{2,2}$")] + val
                    )
                elif column_type == 'enum':
                    column_sql_type = String
                    add_form_extra_fields[column_name] = SelectField(
                        _(column_name),
                        default='',
                        description='',
                        widget=MySelect2Widget(can_input=True,conten2choices=False if columns[column_name].get('choices','') else True),
                        choices=[[x,x] for x in columns[column_name].get('choices','').split(',')]
                    )
                else:
                    add_form_extra_fields[column_name] = StringField(
                        _(column_name),
                        default='',
                        description='',  # columns[column_name]['describe'],
                        widget=BS3TextFieldWidget(),
                        validators=val
                    )

                column_class[column_name] = Column(
                    column_sql_type,
                    nullable=columns[column_name].get('nullable', True),
                    unique=columns[column_name].get('unique', False),
                    primary_key=columns[column_name].get('primary_key', False)
                )

                spec_label_columns[column_name] = columns[column_name]['describe']

                label_columns[column_name] = columns[column_name]['describe']
                description_columns[column_name] = columns[column_name]['describe']
                if not int(columns[column_name].get('primary_key', False)):
                    add_columns.append(column_name)
                show_columns.append(column_name)
                if not int(columns[column_name].get('primary_key', False)):
                    list_columns.append(column_name)
                if column_type == 'string' or column_type == 'text' or column_type == 'int' or column_type == 'enum':
                    if not int(columns[column_name].get('primary_key', False)):
                        search_columns.append(column_name)
                # if column_type == 'int':
                order_columns.append(column_name)

            bind_key = 'dimension_%s' % dim.id
            # SQLALCHEMY_BINDS = conf.get('SQLALCHEMY_BINDS', {})
            # for key in SQLALCHEMY_BINDS:
            conf['SQLALCHEMY_BINDS'][bind_key] = dim.sqllchemy_uri if dim.sqllchemy_uri else default_uri
            # if dim.sqllchemy_uri in SQLALCHEMY_BINDS[key]:
            #     bind_key=key
            #     break
            # model 类
            model_class = type(
                "Dimension_Model_%s" % dim.id, (Model, MyappModelBase),
                dict(
                    __tablename__=dim.table_name,   # 这里表名是不能重复的。不然会相互影响
                    __bind_key__=bind_key if bind_key else None,
                    **column_class
                )
            )

            # 页面视图
            url = '/dimension_remote_table_modelview/%s/api/' % dim_id
            print(url)

            # 代码识别唯一性
            def check_unique(self, item):
                unique_columns = []
                for key in self.cols:
                    if key in ['id']:
                        continue
                    if self.cols[key].get('unique', False):
                        unique_columns.append(key)

                # 查询当前列
                if len(unique_columns)>1:
                    key1 = unique_columns[1]
                    key0 = unique_columns[0]
                    value1={
                        key1:getattr(item,key1)
                    }
                    value0 = {
                        key0: getattr(item, key0)
                    }

                    exist_item = db.session.query(self.datamodel.obj).filter_by(**value1).filter_by(**value0).first()
                    if exist_item:
                        return False
                elif len(unique_columns) > 0:
                    key0 = unique_columns[0]
                    value0 = {
                        key0: getattr(item, key0)
                    }
                    exist_item = db.session.query(self.datamodel.obj).filter_by(**value0).first()
                    if exist_item:
                        return False

                return True


            # 预处理一下
            # @pysnooper.snoop(watch_explode=('item'))
            def pre_add(self, item):

                # 浮点转型
                for key in self.cols:
                    if key in ['id']:
                        continue
                    if self.cols[key].get('column_type', 'text') == 'int':
                        try:
                            setattr(item, key, int(getattr(item, key)))
                        except Exception:
                            setattr(item, key, None)

                    if self.cols[key].get('column_type', 'text') == 'double':
                        try:
                            setattr(item, key, float(getattr(item, key)))
                        except Exception:
                            setattr(item, key, None)
                if not self.check_unique(item):
                    flash(__('检测到唯一性字段重复'), 'warning')
                    raise Exception(__('检测到唯一性字段重复'))

            def pre_update(self, item):

                # 浮点转型
                for key in self.cols:
                    if key in ['id']:
                        continue
                    if self.cols[key].get('column_type', 'text') == 'int':
                        try:
                            setattr(item, key, int(getattr(item, key)))
                        except Exception:
                            setattr(item, key, None)

                    if self.cols[key].get('column_type', 'text') == 'double':
                        try:
                            setattr(item, key, float(getattr(item, key)))
                        except Exception:
                            setattr(item, key, None)
                # 修改必命中唯一性
                # if not self.check_unique(item):
                #     flash(__('检测到唯一性字段重复'), 'warning')
                #     raise Exception(__('检测到唯一性字段重复'))


            def get_primary_key(cols):
                for name in cols:
                    if cols[name].get('primary_key', False):
                        return name
                return ''

            @expose("/upload/", methods=["POST"])
            # @pysnooper.snoop(watch_explode=('attr'))
            def upload(self):
                csv_file = request.files.get('csv_file')  # FileStorage
                dim = db.session.query(Dimension_table).filter_by(id=int(self.dim_id)).first()
                # 文件保存至指定路径
                i_path = csv_file.filename
                if os.path.exists(i_path):
                    os.remove(i_path)
                csv_file.save(i_path)

                if '.xlsx' in i_path:
                    df = pandas.read_excel(i_path, engine='openpyxl')
                    i_path = i_path.replace('.xlsx','.csv')
                    df.to_csv(i_path, index=False)  # index=False 表示不要保存索引

                elif '.xls' in i_path:
                    df = pandas.read_excel(i_path, engine='xlrd')
                    i_path = i_path.replace('.xls', '.csv')
                    df.to_csv(i_path, index=False)  # index=False 表示不要保存索引

                # 读取csv，读取header，按行处理
                import csv

                # 先确定一遍编码
                def get_csv_encoding(file_path, encodings):
                    for encoding in encodings:
                        try:
                            df = pandas.read_csv(file_path, encoding=encoding,header=0)
                            print(f"Successfully read file with {encoding} encoding")
                            return encoding
                        except UnicodeDecodeError:
                            print(f"Failed to read file with {encoding} encoding")
                    message = __('不识别的csv文件编码格式，请转为utf-8编码格式')
                    flash(message,'warning')
                    raise Exception(message)

                encodings = ['utf-8-sig', 'GBK']
                encoding = get_csv_encoding(i_path, encodings)

                csv_reader = csv.reader(open(i_path, mode='r', encoding=encoding))
                header = None
                result = []
                cols = json.loads(dim.columns)
                error_message = []
                for line in csv_reader:
                    if not header:
                        header = line
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
                                return self.response(400, **back)
                        continue
                    # 个数不对的去掉
                    if len(line) != len(header):
                        continue

                    # 全是空值的去掉
                    ll = [l.strip() for l in line if l.strip()]
                    if not ll:
                        continue

                    data = dict(zip(header, line))

                    try:
                        # 把整型做一下转换，因为文件离线全部识别为字符串
                        for key in copy.deepcopy(data):
                            try:
                                if cols.get(key, {}).get('column_type', 'text') == 'int':
                                    data[key] = int(data[key])
                                elif cols.get(key, {}).get('column_type', 'text') == 'double':
                                    data[key] = float(data[key]) if data[key] else None
                                else:
                                    data[key] = str(data[key]).replace('\n', ' ')
                            except Exception as e:
                                print(e)
                                data[key] = None

                        model = self.datamodel.obj(**data)
                        self.pre_add(model)
                        db.session.add(model)
                        self.post_add(model)
                        db.session.commit()
                        result.append('success')
                    # except SQLAlchemyError as ex:
                    #     db.session.rollback()
                    except Exception as e:
                        db.session.rollback()
                        print(e)
                        error_message.append(str(e))
                        result.append(str(e) + "-----------")

                # flash('成功导入%s行，失败导入%s行' % (len([x for x in result if x == 'success']), len([x for x in result if x == 'fail'])), 'success')

                # flash('上传失败%s'%error_message,'error')
                # back = {
                #     "status": 0,
                #     "result": result,
                #     "message": "result为上传成功行，共成功%s" % len([x for x in result if x == 'success'])
                # }
                # return self.response(200, **back)
                message = 'success %s rows，fail %s rows' % (len([x for x in result if x == 'success']), len([x for x in result if x != 'success']))
                message += ','.join(result)
                message = Markup(message)
                return make_response(message, 200)

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

            @action("copy_row", "复制", "复制所选记录?", "fa-trash", single=False)
            # @pysnooper.snoop(watch_explode=('items'))
            def copy_row(self, items):
                if not items:
                    abort(404)
                success = []
                fail = []
                for item in items:
                    try:
                        req_json = item.to_json()
                        if 'id' in req_json:
                            del req_json["id"]
                        json_data = self.pre_add_req(req_json)
                        new_item = self.add_model_schema.load(json_data)
                        self.pre_add(new_item.data)
                        self.datamodel.add(new_item.data, raise_exception=True)
                        self.post_add(new_item.data)
                        result_data = self.add_model_schema.dump(new_item.data, many=False).data
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

            view_class = type(
                "Dimension_%s_ModelView_Api" % dim.id, (MyappModelRestApi,),
                dict(
                    datamodel=SQLAInterface(model_class, session=db.session),
                    route_base=url,
                    add_form_extra_fields=add_form_extra_fields,
                    edit_form_extra_fields=add_form_extra_fields,
                    spec_label_columns=spec_label_columns,
                    search_columns=search_columns,
                    order_columns=order_columns,
                    add_columns=add_columns,
                    list_columns=list_columns,
                    label_title=dim.label,
                    base_permissions=['can_list', 'can_add', 'can_delete', 'can_edit', 'can_show'],
                    pre_add=pre_add,
                    pre_update=pre_update,
                    upload=upload,
                    muldelete=muldelete,
                    check_unique=check_unique,
                    copy_row=copy_row,
                    dim_id=dim_id,
                    import_data=True,
                    download_data=True,
                    cols_width=cols_width,
                    base_order=(get_primary_key(columns), "desc") if get_primary_key(columns) else None,
                    cols=columns
                )
            )
            view_instance = view_class()
            view_instance._init_model_schemas()
            all_dimension["dimension_%s" % dim_id] = view_instance

        return all_dimension["dimension_%s" % dim_id]

    @expose("/<dim_id>/api/_info", methods=["GET"])
    def dim_api_info(self, dim_id, **kwargs):
        view_instance = self.set_model(dim_id)
        return view_instance.api_info(**kwargs)

    @expose("/<dim_id>/api/<int:pk>", methods=["GET"])
    def dim_api_show(self, dim_id, pk, **kwargs):
        view_instance = self.set_model(dim_id)
        return view_instance.api_show(pk, **kwargs)

    @expose("/<dim_id>/api/", methods=["GET"])
    def dim_api_list(self, dim_id, **kwargs):
        view_instance = self.set_model(dim_id)
        try:
            return view_instance.api_list(**kwargs)
        except Exception as e:
            print(e)

            # 更新以后表结构会变
            all_dimension = conf.get('all_dimension_instance', {})
            if "dimension_%s" % dim_id in all_dimension:
                del all_dimension["dimension_%s" % dim_id]
            flash(Markup(str(e)),'error')
            return jsonify({
                "status": 1,
                "message": str(e),
                "result": ""
            })

    @expose("/<dim_id>/api/", methods=["POST"])
    def dim_api_add(self, dim_id):
        view_instance = self.set_model(dim_id)

        try:
            return view_instance.api_add()
        except Exception as e:
            print(e)
            flash(Markup(str(e)), 'error')
            return jsonify({
                "status": 1,
                "message": str(e),
                "result": ""
            })

    @expose("/<dim_id>/api/<pk>", methods=["PUT"])
    # @pysnooper.snoop(watch_explode=('item','data'))
    def dim_api_edit(self, dim_id, pk):
        view_instance = self.set_model(dim_id)
        return view_instance.api_edit(pk)

    @expose("/<dim_id>/api/<pk>", methods=["DELETE"])
    def dim_api_delete(self, dim_id, pk):
        view_instance = self.set_model(dim_id)
        return view_instance.api_delete(pk)

    @expose("/<dim_id>/api/upload/", methods=["POST"])
    # @pysnooper.snoop()
    def dim_api_upload(self, dim_id):
        view_instance = self.set_model(dim_id)
        return view_instance.upload()

    @expose("/<dim_id>/api/download_template", methods=["GET"])
    @expose("/<dim_id>/api/download_template/", methods=["GET"])
    # @pysnooper.snoop()
    def dim_api_download_template(self, dim_id):
        view_instance = self.set_model(dim_id)
        return view_instance.download_template()

    @expose("/<dim_id>/api/download/", methods=["GET"])
    # @pysnooper.snoop()
    def dim_api_download(self, dim_id):
        view_instance = self.set_model(dim_id)
        return view_instance.download()

    @expose("/<dim_id>/api/multi_action/<string:name>", methods=["POST"])
    def multi_action(self, dim_id, name):
        view_instance = self.set_model(dim_id)
        return view_instance.multi_action(name)


appbuilder.add_api(Dimension_remote_table_ModelView_Api)
