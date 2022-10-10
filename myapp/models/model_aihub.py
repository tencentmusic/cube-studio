from flask_appbuilder import Model

from sqlalchemy import (
    Boolean,
    Column,
    create_engine,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Enum,
)

from myapp import app,db
from myapp.models.helpers import ImportMixin
# from myapp.models.base import MyappModel
from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime

from flask import Markup
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config


class Aihub(Model,ImportMixin,MyappModelBase):
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


    @property
    def card(self):
        notebook_url = "/aihub/api/notebook/"+self.uuid if self.status=='online' and self.notebook else ""
        train_url = "/aihub/api/train/" + self.uuid if self.status == 'online' and (self.job_template or self.pipeline) else ""
        service_url = "/aihub/api/service/" + self.uuid if self.status == 'online' and (self.service or self.inference) else ""

        return Markup(f'''
<div style="border: 3px solid rgba({'29,152,29,.9' if self.status=='online' else '0,0,0,.2'});border-radius: 3px;">
    <a href="{self.doc if self.status=='online' else ''}">
        <img src="{self.pic}" style="height:200px;width:100%" alt="{self.describe}"/>
    </a>
    <br>
    <div>
        <div class="p16" alt="{self.describe}">
            <div class="p-r card-popup ellip1">
                {("在线:" if self.status=='online' else '待上线:')+self.describe}
                <div class="p-a card-popup-target d-n" style="top:100%;left:0;background:rgba(0,0,0,0.5);color:#fff;border-radius:3px;">{self.describe}</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(0,0,0,.06);" class="ptb8 d-f ac jc-b">
            <a class="flex1 ta-c" style="border-right: 1px solid rgba(0,0,0,.06);" href='{notebook_url}'>notebook</a>
            <a class="flex1 ta-c" style="border-right: 1px solid rgba(0,0,0,.06);" href='{train_url}'>训练</a>
            <a class="flex1 ta-c" style="border-right: 1px solid rgba(0,0,0,.06);" href='{service_url}'>服务</a>
        </div>
    </div>
</div>
''')




