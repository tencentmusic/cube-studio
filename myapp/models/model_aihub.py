import json
import os

from flask_appbuilder import Model

from sqlalchemy import Text

from myapp import app
from myapp.models.helpers import ImportMixin
from sqlalchemy import Column, Integer, String

from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Aihub(Model,MyappModelBase):
    __tablename__ = 'aihub'

    id = Column(Integer, primary_key=True)
    uuid = Column(String(2000), nullable=True, default='')
    status = Column(String(200), nullable=True, default='')
    doc = Column(String(200), nullable=True,default='')
    name = Column(String(200), nullable=True, default='')
    field = Column(String(200), nullable=True, default='')
    scenes = Column(String(200), nullable=True, default='')
    type = Column(String(200), nullable=True, default='')
    label = Column(String(200), nullable=True, default='')
    describe = Column(String(2000), nullable=True, default='')
    source = Column(String(200), nullable=True, default='')
    pic = Column(String(500), nullable=True, default='')
    images = Column(String(200), nullable=True, default='')
    dataset = Column(Text, nullable=True, default='')
    notebook = Column(String(2000), nullable=True, default='')
    job_template = Column(Text, nullable=True, default='')
    pipeline = Column(Text, nullable=True, default='')
    pre_train_model = Column(String(2000), nullable=True, default='')
    inference = Column(Text, nullable=True, default='')
    service = Column(Text, nullable=True, default='')
    version = Column(String(200), nullable=True, default='')
    hot = Column(Integer, default=1)
    price = Column(Integer, default=1)
    expand = Column(Text, nullable=True, default='{}')


    @property
    def card(self):

        pic_url = '/static/aihub/deep-learning/'+self.name+'/'+self.pic

        if 'http://' in self.pic or "https://" in self.pic:
            pic_url=self.pic

        ops_html=f'''
            <a class="flex1 ta-c" target=_blank style="color:Gray;border-right: 1px solid rgba(0,0,0,.06);" href='javascript:void(0)'>开发</a>
            <a class="flex1 ta-c" target=_blank style="color:Gray;border-right: 1px solid rgba(0,0,0,.06);" href="javascript:void(0)">训练</a>
            <a class="flex1 ta-c" target=_blank style="color:Gray;border-right: 1px solid rgba(0,0,0,.06);" href='javascript:void(0)'>部署web</a>
        '''

        if self.price and int(self.price)>0:
            ops_html = f'''
            <a class="flex1 ta-c" style="color:Gray;border-right: 1px solid rgba(0,0,0,.06);" href="javascript:void(0)">企业版</a>
            '''

        return Markup(f'''
<div style="border: 3px solid rgba({'29,152,29,.6' if self.status=='online' else '0,0,0,.2'});border-radius: 3px;">
        <img src="{pic_url}" onerror="this.src='/static/assets/images/aihub_loading.gif'" style="height:200px;width:100%" alt="{self.describe}"/>
    <br>
    <div>
        <div class="p16" alt="{self.describe}">
            <div class="p-r card-popup ellip1">
                { self.label }
                <div class="p-a card-popup-target d-n" style="top:100%;left:0;background:rgba(0,0,0,0.5);color:#fff;border-radius:3px;">{self.describe}</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(0,0,0,.06);" class="ptb8 d-f ac jc-b">
{ops_html}
        </div>

    </div>
</div>
''')




