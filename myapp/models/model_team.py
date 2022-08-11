from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey,Float
from sqlalchemy.orm import relationship
import datetime,time,json
from myapp import app,db
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
    Enum
)
from myapp.models.helpers import AuditMixinNullable, ImportMixin
from sqlalchemy.orm import backref, relationship
from myapp.models.base import MyappModelBase
from myapp.models.helpers import ImportMixin

from sqlalchemy import Column, Integer, String, ForeignKey ,Date,DateTime
from flask_appbuilder.models.decorators import renders
from flask import Markup
import datetime
from sqlalchemy import String,Column,Integer,ForeignKey,UniqueConstraint
from myapp.security import MyUser
metadata = Model.metadata
conf = app.config
import pysnooper



class Project(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'project'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    describe = Column(String(500), nullable=False)
    type = Column(String(50))   # org, job_template, model
    expand = Column(Text(65536), default='{}')

    export_children = ["user"]

    __table_args__ = (
        UniqueConstraint('name','type'),
    )

    def __repr__(self):
        return self.name

    @renders('expand')
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    def get_creators(self):
        creators = db.session().query(Project_User).filter_by(project_id=self.id).all()
        if creators:
            return [creator.user.username for creator in creators]
        else:
            return []

    @property
    def node_selector(self):
        try:
            expand = json.loads(self.expand) if self.expand else {}
            node_selector = expand.get('node_selector', '')
            return node_selector
        except Exception as e:
            print(e)
            return ''

    @property
    def volume_mount(self):
        try:
            expand = json.loads(self.expand) if self.expand else {}
            volume_mount = expand.get('volume_mount', '')
            if not volume_mount:
                volume_mount='kubeflow-user-workspace(pvc):/mnt,kubeflow-archives(pvc):/archives'
            return volume_mount
        except Exception as e:
            print(e)
            return ''

    @property
    def cluster(self):
        all_clusters = conf.get('CLUSTERS')
        default=all_clusters.get(conf.get('ENVIRONMENT'),{})
        try:
            expand = json.loads(self.expand) if self.expand else {}
            project_cluster = expand.get('cluster','')
            if project_cluster and project_cluster in all_clusters:
                return all_clusters[project_cluster]
            return default
        except Exception as e:
            print(e)
            return default

class Project_User(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = "project_user"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("project.id"),nullable=False)
    project = relationship(
        "Project",
        backref=backref("user", cascade="all, delete-orphan"),
        foreign_keys=[project_id],
    )
    user_id = Column(Integer, ForeignKey("ab_user.id"),nullable=False)
    user = relationship(
        "MyUser",
        backref=backref("user", cascade="all, delete-orphan"),
        foreign_keys=[user_id],
    )
    role = Column(Enum('dev', 'ops','creator'),nullable=False,default='dev')
    # role = Column(String(50), nullable=False, default='dev')
    export_parent = "project"

    def __repr__(self):
        return self.user.username+"(%s)"%self.role

    __table_args__ = (
        UniqueConstraint('project_id','user_id'),
    )
