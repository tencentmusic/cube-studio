import json
import os
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder import Model

from sqlalchemy import Text

from myapp import app
from sqlalchemy import Column, Integer, String

from flask import Markup,request
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config


class Aihub(Model,MyappModelBase):
    __tablename__ = 'aihub'

    id = Column(Integer, primary_key=True,comment='id主键')
    uuid = Column(String(2000), nullable=True, default='',comment='当前版本的唯一主键')
    status = Column(String(200), nullable=True, default='',comment='状态')
    doc = Column(String(200), nullable=True,default='',comment='文档地址')
    name = Column(String(200), nullable=True, default='',comment='英文名')
    field = Column(String(200), nullable=True, default='',comment='领域')
    scenes = Column(String(200), nullable=True, default='',comment='应用场景')
    type = Column(String(200), nullable=True, default='',comment='类型')
    label = Column(String(200), nullable=True, default='',comment='中文名')
    describe = Column(String(2000), nullable=True, default='',comment='描述')
    source = Column(String(200), nullable=True, default='',comment='来源')
    pic = Column(String(500), nullable=True, default='',comment='示例图片')
    images = Column(String(200), nullable=True, default='',comment='docker镜像')
    dataset = Column(Text, nullable=True, default='',comment='数据集的配置')
    notebook = Column(String(2000), nullable=True, default='',comment='notebook的配置')
    job_template = Column(Text, nullable=True, default='',comment='任务模板参数')
    pipeline = Column(Text, nullable=True, default='',comment='pipeline微调的配置')
    pre_train_model = Column(String(2000), nullable=True, default='',comment='预训练模型地址')
    inference = Column(Text, nullable=True, default='',comment='推理服务的配置')
    service = Column(Text, nullable=True, default='',comment='服务化参数')
    version = Column(String(200), nullable=True, default='',comment='版本')
    hot = Column(Integer, default=1,comment='热度')
    price = Column(Integer, default=1,comment='价格')
    expand = Column(Text, nullable=True, default='{}',comment='扩展参数')


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




