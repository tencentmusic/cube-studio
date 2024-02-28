import pysnooper
from flask_appbuilder import Model
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from sqlalchemy import Text
import os,time,json
from myapp.models.helpers import AuditMixinNullable
from myapp import app
from sqlalchemy import Column, Integer, String
from flask import Markup,request,g
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config



class Dataset(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'dataset'
    id = Column(Integer, primary_key=True,comment='id主键')
    name =  Column(String(200), nullable=True,comment='英文名')  #
    label = Column(String(200), nullable=True,comment='中文名')  #
    describe = Column(String(2000), nullable=True,comment='描述') #
    version = Column(String(200), nullable=True, default='',comment='版本')  #
    subdataset = Column(String(200), nullable=True, default='',comment=' 数据子集名称，例如英文数据子集，中文数据子集')  #
    split = Column(String(200), nullable=True, default='',comment=' train test val等')  #
    segment = Column(Text, nullable=True, default='{}',comment='可以追加数据块，避免整块更新，记录分区信息。分区名，文件文件信息')  #

    doc = Column(String(200), nullable=True, default='',comment='数据集的文档页面')  #
    source_type = Column(String(200), nullable=True,comment='数据集来源，开源，资产，采购')  #
    source = Column(String(200), nullable=True,comment='数据集来源，github, 天池')  #
    industry =  Column(String(200), nullable=True,comment='行业')  #
    icon = Column(String(2000), nullable=True,comment='图标svg内容')  #
    field = Column(String(200), nullable=True,comment='数据领域，视觉，听觉，文本')  #
    usage = Column(String(200), nullable=True,comment='数据用途')  #
    research = Column(String(200), nullable=True,comment='研究方向')  #

    storage_class = Column(String(200), nullable=True, default='',comment='存储类型，压缩') #
    file_type = Column(String(200), nullable=True,default='',comment='文件类型，图片 png，音频')  #
    status = Column(String(200), nullable=True, default='',comment='文件类型  有待校验，已下线')  #

    years = Column(String(200), nullable=True,comment='年份')  #

    url = Column(String(1000),nullable=True,comment='关联url')  #
    path = Column(String(400),nullable=True,comment='本地的持久化路径')  #
    download_url = Column(String(1000),nullable=True,comment='下载地址')  #
    storage_size = Column(String(200), nullable=True,default='',comment='存储大小')  #
    entries_num = Column(String(200), nullable=True, default='',comment='记录数目')  #
    duration = Column(String(200), nullable=True, default='',comment='时长')  #
    price = Column(String(200), nullable=True, default='0',comment='价格')  #

    secret = Column(String(200), nullable=True, default='',comment='秘钥，数据集的秘钥')  #
    info = Column(Text, nullable=True,default='{}',comment='数据集，内容信息')  #
    features = Column(Text, nullable=True,default='{}',comment='特征信息')  #
    metric_info = Column(Text, nullable=True,default='{}',comment='数据集，指标信息')  #

    owner = Column(String(200),nullable=True,default='*',comment='责任人，*表示全部可见')  #

    expand = Column(Text(65536), nullable=True,default='{}',comment='扩展参数')

    def __repr__(self):
        return self.name

    @property
    def url_html(self):
        urls= self.url.split('\n')

        html = ''
        for url in urls:
            if url.strip():
                html+='<a target=_blank href="%s">%s</a><br>'%(url.strip(),url.strip())
        return Markup('<div>%s</div>'%html)

    @property
    def path_html(self):
        paths= self.path.split('\n')

        html = ''
        for path in paths:
            if path.strip():
                download_url = request.host_url+'/static/'+path.replace('/data/k8s/kubeflow/','')
                html += f'<a target=_blank href="{download_url}">{path.strip()}</a><br>'
        return Markup('<div>%s</div>'%html)


    @property
    def icon_html(self):
        img_url = self.icon if self.icon else "/static/assets/images/dataset.png"

        url = f'''
<a target=_blank href='{img_url}'>
  <img style='height:50px; width:50px; border-radius:10%' src='{img_url}'>
</a>
        '''
        # print(url)
        return url

    @property
    def download_url_html(self):
        urls= self.download_url.split('\n')
        html = ''
        for url in urls:
            if url.strip():
                html += '<a target=_blank href="%s">%s</a><br>' % (url.strip(), url.strip())
        return Markup('<div>%s</div>'%html)

    @property
    def ops_html(self):
        if '*' in self.owner or g.user.username in self.owner:
            if self.path and '.csv' in self.path:
                dom = f'''
                <a target=_blank href="/dataset_modelview/api/explore/{self.id}">{__("探索")}</a>
                '''
                return Markup(dom)

        return __("探索")

    @property
    # @pysnooper.snoop()
    def card(self):
        label_url = f"/dataset_modelview/api/label/{self.id}"
        train_url = f"/dataset_modelview/api/train/{self.id}"

        ops_html = f'''
            <a class="flex1 ta-c" target=_blank style="border-right: 1px solid rgba(0,0,0,.06);" href='{label_url}'>{__("标注")}</a>
            <a class="flex1 ta-c" target=_blank style="border-right: 1px solid rgba(0,0,0,.06);" href="{train_url}">{__("计算")}</a>
        '''
        describe = self.label+f"({self.describe})"
        return Markup(f'''
<div style="border: 3px solid rgba({'29,152,29,.6' if self.status == 'online' else '0,0,0,.2'});border-radius: 3px;">

        <img src="{self.icon}" onerror="this.src='/static/assets/images/aihub_loading.gif'" style="height:200px;width:100%" alt="{self.describe}"/>

    <br>
    <div>
        <div class="p16" alt="{self.describe}">
            <div class="p-r card-popup ellip1">
                {describe}
                <div class="p-a card-popup-target d-n" style="top:100%;left:0;background:rgba(0,0,0,0.5);color:#fff;border-radius:3px;">{self.describe}</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(0,0,0,.06);" class="ptb8 d-f ac jc-b">
{ops_html}
        </div>
    </div>
</div>
''')