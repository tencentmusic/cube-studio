from flask_appbuilder import Model
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask import g
from myapp import app
from myapp.models.helpers import ImportMixin
from sqlalchemy import Column, Integer, String
from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config



class Dimension_table(Model,ImportMixin,MyappModelBase):
    __tablename__ = 'dimension'
    __table_args__ = (UniqueConstraint('sqllchemy_uri', 'table_name'),)
    id = Column(Integer, primary_key=True,comment='id主键')
    sqllchemy_uri = Column(String(255),nullable=True,comment='sql链接串')
    table_name = Column(String(255),nullable=True,unique=True,comment='表名')
    label = Column(String(255), nullable=True,comment='中文名')
    describe = Column(String(2000), nullable=True,comment='描述')
    app = Column(String(255), nullable=True,comment='应用名')
    owner = Column(String(2000), nullable=True,default='',comment='责任人')
    columns=Column(Text, nullable=True,default='{}',comment='列信息')
    status = Column(Integer, default=1,comment='状态')


    @property
    def table_html(self):
        users=''
        users = users+self.owner if self.owner else users
        users = users.split(',')
        users = [user.strip() for user in users if user.strip()]
        url_path = conf.get('MODEL_URLS',{}).get("dimension")
        if g.user.is_admin() or g.user.username in users or '*' in self.owner:
            if self.sqllchemy_uri:
                return Markup(f'<a target=_blank href="{url_path}?targetId={self.id}">{self.table_name}</a>')

        return self.table_name

    @property
    def operate_html(self):
        if self.sqllchemy_uri:
            url=f'''
            <a target=_blank href="/dimension_table_modelview/api/create_external_table/%s">{__("更新远程表")}</a>  | <a target=_blank href="/dimension_table_modelview/api/external/%s">{__("建外表示例")}</a>
            '''%(self.id,self.id)
            return Markup(url)
        else:
            return __("更新远程表")+" | "+__("建外表示例")






