import re
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder import Model
import json
from myapp import app,db
from sqlalchemy import (
    Text,
    Enum
)
from myapp.models.helpers import AuditMixinNullable
from sqlalchemy.orm import backref, relationship
from myapp.models.base import MyappModelBase

from flask_appbuilder.models.decorators import renders
from flask import Markup
from sqlalchemy import String,Column,Integer,ForeignKey,UniqueConstraint



metadata = Model.metadata
conf = app.config
import pysnooper


class Project(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = 'project'
    id = Column(Integer, primary_key=True,comment='id主键')
    name = Column(String(50), nullable=False,comment='名称')
    describe = Column(String(500), nullable=False,comment='描述')
    type = Column(String(50),comment='类型，org, job_template, model')   #
    expand = Column(Text(65536), default='{}',comment='扩展参数')

    export_children = ["user"]

    __table_args__ = (
        UniqueConstraint('name','type'),
    )

    def __repr__(self):
        return self.name

    @property
    def expand_html(self):
        return Markup('<pre><code>' + self.expand + '</code></pre>')

    # 获取用户角色
    def user_role(self,user_id):
        project_user = db.session().query(Project_User).filter_by(project_id=self.id).filter_by(user_id=user_id).first()
        if project_user:
            return project_user.role
        return ''


    def get_creators(self):
        creators = db.session().query(Project_User).filter_by(project_id=self.id).filter_by(role='creator').all()
        if creators:
            return [creator.user.username for creator in creators]
        else:
            return []

    @property
    def node_selector(self):
        try:
            expand = json.loads(self.expand) if self.expand else {}
            node_selector = expand.get('node_selector', '')
            if 'org' in expand:
                node_selector+=',org='+expand['org']
            node_selector = ','.join(list(set([x for x in node_selector.split(',') if x])))
            return node_selector
        except Exception as e:
            print(e)
            return ''

    @property
    def volume_mount(self):
        try:
            expand = json.loads(self.expand) if self.expand else {}
            volume_mount = expand.get('volume_mount', '')

            # 如果不包含/mnt和/archives就加上，这两个目录是固定个人子目录
            volume_mount_list=re.split(',|;|\n|\t',volume_mount)
            no_user_workspace = True
            no_user_archives = True
            for volume in volume_mount_list:
                if volume[-4:]=='/mnt' or volume[-5:]=='/mnt/':
                    no_user_workspace = False
                if volume[-9:]=='/archives' or volume[-10:]=='/archives/':
                    no_user_archives = False
            if no_user_workspace:
                volume_mount_list.append('kubeflow-user-workspace(pvc):/mnt')
            if no_user_workspace and no_user_archives:
                volume_mount_list.append('kubeflow-archives(pvc):/archives')
            volume_mount=','.join(list(set(volume_mount_list)))
            volume_mount = volume_mount.strip(',')
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

    @property
    def cluster_name(self):
        return self.cluster['NAME']

    @property
    def org(self):
        expand = json.loads(self.expand) if self.expand else {}
        return expand.get('org','public')

    @property
    def job_template(self):
        try:
            from myapp.models.model_job import Job_Template
            job_templates = db.session.query(Job_Template.name).filter_by(project_id=self.id).all()
            return ','.join([x[0] for x in job_templates])
        except Exception as e:
            print(e)

        return ''

    @property
    def project_user(self):
        try:
            project_user = db.session.query(Project_User).filter_by(project_id=self.id).all()
            return ','.join([str(x) for x in project_user])
        except Exception as e:
            print(e)
            return ''

class Project_User(Model,AuditMixinNullable,MyappModelBase):
    __tablename__ = "project_user"
    id = Column(Integer, primary_key=True,comment='id主键')
    project_id = Column(Integer, ForeignKey("project.id"),nullable=False,comment='项目组id')
    project = relationship(
        "Project",
        backref=backref("user", cascade="all, delete-orphan"),
        foreign_keys=[project_id],
    )
    user_id = Column(Integer, ForeignKey("ab_user.id"),nullable=False,comment='用户id')
    user = relationship(
        "MyUser",
        backref=backref("user", cascade="all, delete-orphan"),
        foreign_keys=[user_id],
    )
    role = Column(Enum('dev', 'ops','creator',name='role'),nullable=False,default='dev',comment='角色')
    quota = Column(String(2000), nullable=True, default='',comment='用户在该项目中可以使用的资源额度')
    export_parent = "project"

    def __repr__(self):
        return self.user.username+f"({self.role})"

    __table_args__ = (
        UniqueConstraint('project_id','user_id'),
    )

