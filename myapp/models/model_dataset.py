from flask_appbuilder import Model

from sqlalchemy import Text

from myapp.models.helpers import AuditMixinNullable
from myapp import app
from sqlalchemy import Column, Integer, String
from flask import Markup
from myapp.models.base import MyappModelBase
metadata = Model.metadata
conf = app.config



class Dataset(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'dataset'
    id = Column(Integer, primary_key=True)
    name =  Column(String(200), nullable=True)  #
    label = Column(String(200), nullable=True)  #
    describe = Column(String(2000), nullable=True) #

    source_type = Column(String(200), nullable=True)  # 数据集来源，开源，资产，采购
    source = Column(String(200), nullable=True)  # 数据集来源，github, 天池
    industry =  Column(String(200), nullable=True)  # 行业，
    icon = Column(String(2000), nullable=True)  # 图标
    field =  Column(String(200), nullable=True)  # 数据领域，视觉，听觉，文本
    usage =  Column(String(200), nullable=True)  # 数据用途
    research =  Column(String(200), nullable=True)  # 研究方向

    storage_class = Column(String(200), nullable=True, default='') # 存储类型，压缩
    file_type = Column(String(200), nullable=True,default='')  # 文件类型，图片 png，音频
    status = Column(String(200), nullable=True, default='')  # 文件类型  有待校验，已下线

    years = Column(String(200), nullable=True)  # 年份
    url  = Column(String(1000),nullable=True)  # 关联url
    path = Column(String(400),nullable=True)  # 本地的持久化路径
    download_url = Column(String(1000),nullable=True)  # 下载地址
    storage_size = Column(String(200), nullable=True,default='')  # 存储大小
    entries_num = Column(String(200), nullable=True, default='')  # 记录数目
    duration = Column(String(200), nullable=True, default='')  # 时长
    price = Column(String(200), nullable=True, default='0')  # 价格



    owner = Column(String(200),nullable=True)  #

    expand = Column(Text(65536), nullable=True,default='{}')

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
    def icon_html(self):
        img_url = self.icon if self.icon else "/static/assets/images/dataset.png"
        url = '<a target=_blank href="%s"><img height="50px" width="50px" src="%s"></a>' % (img_url,img_url)
        return Markup(url)

    @property
    def download_url_html(self):
        urls= self.download_url.split('\n')
        html = ''
        for url in urls:
            if url.strip():
                html += '<a target=_blank href="%s">%s</a><br>' % (url.strip(), url.strip())
        return Markup('<div>%s</div>'%html)
