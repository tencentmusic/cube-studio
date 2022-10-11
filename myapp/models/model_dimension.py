from flask_appbuilder import Model
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint

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
    id = Column(Integer, primary_key=True)
    sqllchemy_uri = Column(String(255),nullable=True)
    table_name = Column(String(255),nullable=True,unique=True)
    label = Column(String(255), nullable=True)
    describe = Column(String(2000), nullable=True)
    app = Column(String(255), nullable=True)
    owner = Column(String(2000), nullable=True,default='')
    columns=Column(Text, nullable=True,default='{}')
    status = Column(Integer, default=1)


    @property
    def table_html(self):
        users=''
        users = users+self.owner if self.owner else users
        users = users.split(',')
        users = [user.strip() for user in users if user.strip()]
        url_path = conf.get('MODEL_URLS',{}).get("dimension")
        if g.user.is_admin() or g.user.username in users or '*' in self.owner:
            return Markup(f'<a target=_blank href="{url_path}?targetId={self.id}">{self.table_name}</a>')
        else:
            return self.table_name

    @property
    def operate_html(self):
        url='''
        <a target=_blank href="/dimension_table_modelview/api/create_external_table/%s">创建远程表</a>  | <a target=_blank href="/dimension_table_modelview/api/external/%s">建外表示例</a> | <a href="/dimension_table_modelview/api/clear/%s">清空表记录</a>
        '''%(self.id,self.id,self.id)
        return Markup(url)





