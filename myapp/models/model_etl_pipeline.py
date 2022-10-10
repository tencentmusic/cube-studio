from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey,Float
from sqlalchemy.orm import relationship
import datetime,time,json
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

from myapp.models.helpers import AuditMixinNullable, ImportMixin

from myapp import app,db
from myapp.models.helpers import ImportMixin
# from myapp.models.base import MyappModel

from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
from myapp.models.base import MyappModelBase
import datetime
metadata = Model.metadata
conf = app.config



class ETL_Pipeline(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'etl_pipeline'
    id = Column(Integer, primary_key=True)
    name = Column(String(100),nullable=False,unique=True)
    describe = Column(String(200),nullable=False)
    project_id = Column(Integer, ForeignKey('project.id'),nullable=False)  # 定义外键
    project = relationship(
        "Project", foreign_keys=[project_id]
    )
    dag_json = Column(Text(65536),nullable=True,default='{}')  # pipeline的上下游关系
    config = Column(Text(65536),default='{}')   # pipeline的全局配置
    expand = Column(Text(65536),default='[]')

    # export_children = "etl_task"

    def __repr__(self):
        return self.name

    @property
    def etl_pipeline_url(self):
        pipeline_url="/etl_pipeline_modelview/web/" +str(self.id)
        return Markup(f'<a target=_blank href="{pipeline_url}">{self.describe}</a>')


    @renders('dag_json')
    def dag_json_html(self):
        dag_json = self.dag_json or '{}'
        return Markup('<pre><code>' + dag_json + '</code></pre>')


    @renders('config')
    def config_html(self):
        config = self.config or '{}'
        return Markup('<pre><code>' + config + '</code></pre>')

    @renders('expand')
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    @renders('parameter')
    def parameter_html(self):
        return Markup('<pre><code>' + self.parameter + '</code></pre>')


    @property
    def run_instance(self):
        # workflow = db.session.query(Workflow).filter_by(foreign_key= str(self.id)).filter_by(status= 'Running').filter_by(create_time > datetime.datetime.now().strftime("%Y-%m-%d")).all()
        # workflow_num = len(workflow) if workflow else 0
        # url = '/workflow_modelview/list/?_flt_2_name=%s'%self.name.replace("_","-")[:54]
        url_path = conf.get('MODEL_URLS', {}).get("etl_task_instance")
        # print(url)
        return Markup(f"<a target=_blank href='{url_path}'>任务实例</a>")


    def clone(self):
        return ETL_Pipeline(
            name=self.name.replace('_', '-'),
            describe=self.describe,
            project_id=self.project_id,
            dag_json=self.dag_json,
            config=self.config,
            expand=self.expand,
        )


class ETL_Task(Model,ImportMixin,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'etl_task'
    id = Column(Integer, primary_key=True)
    name = Column(String(100),nullable=False,unique=True)
    describe = Column(String(200),nullable=False)
    etl_pipeline_id = Column(Integer, ForeignKey('etl_pipeline.id'),nullable=False)  # 定义外键
    etl_pipeline = relationship(
        "ETL_Pipeline", foreign_keys=[etl_pipeline_id]
    )
    template = Column(String(100),nullable=False)
    task_args = Column(Text(65536),default='{}')
    etl_task_id = Column(String(100),nullable=False)
    expand = Column(Text(65536),default='[]')


    # export_parent = "etl_pipeline"


    def __repr__(self):
        return self.name

    @property
    def etl_pipeline_url(self):
        pipeline_url="/etl_pipeline_modelview/web/" +str(self.etl_pipeline.id)
        return Markup(f'<a target=_blank href="{pipeline_url}">{self.etl_pipeline.describe}</a>')



