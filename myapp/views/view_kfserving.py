from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask import Blueprint, current_app, jsonify, make_response, request
# 将model添加成视图，并控制在前端的显示
from myapp.models.model_serving import Service,KfService
from myapp.models.model_team import Project,Project_User
from myapp.utils import core
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from flask_appbuilder.actions import action
from myapp import app, appbuilder,db,event_logger
import logging
import re
import uuid
import requests
from myapp.exceptions import MyappException
from flask_appbuilder.security.decorators import has_access
from myapp.models.model_job import Repository
from flask_wtf.file import FileAllowed, FileField, FileRequired
from werkzeug.datastructures import FileStorage
from wtforms.ext.sqlalchemy.fields import QuerySelectField
from myapp import security_manager
import os,sys
from wtforms.validators import DataRequired, Length, NumberRange, Optional,Regexp
from wtforms import BooleanField, IntegerField, SelectField, StringField,FloatField,DateField,DateTimeField,SelectMultipleField,FormField,FieldList
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget,BS3PasswordFieldWidget,DatePickerWidget,DateTimePickerWidget,Select2ManyWidget,Select2Widget
from myapp.forms import MyBS3TextAreaFieldWidget,MySelect2Widget,MyCodeArea,MyLineSeparatedListField,MyJSONField,MyBS3TextFieldWidget,MySelectMultipleField
from myapp.utils.py import py_k8s
import os, zipfile
import shutil
from flask import (
    current_app,
    abort,
    flash,
    g,
    Markup,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)
from .base import (
    DeleteMixin,
    api,
    BaseMyappView,
    check_ownership,
    CsvResponse,
    data_payload_response,
    DeleteMixin,
    generate_download_headers,
    get_error_msg,
    get_user_roles,
    handle_api_exception,
    json_error_response,
    json_success,
    MyappFilter,
    MyappModelView,

)
from sqlalchemy import and_, or_, select
from .baseApi import (
    MyappModelRestApi
)

import kubernetes
from kfserving import KFServingClient
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2CustomSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2TensorflowSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService

from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config


class KfService_ModelView(MyappModelView):
    datamodel = SQLAInterface(KfService)
    crd_name = 'inferenceservice'
    help_url = conf.get('HELP_URL', {}).get(datamodel.obj.__tablename__, '') if datamodel else ''
    show_columns = ['name', 'label','service_type','default_service','canary_service','canary_traffic_percent','k8s_yaml']
    add_columns = ['name', 'label', 'service_type','default_service','canary_service','canary_traffic_percent']
    list_columns = ['label_url','host','service','deploy','status','roll']
    edit_columns = add_columns
    base_order = ('id','desc')
    order_columns = ['id']

    @expose('/deploy1/<kfservice_id>',methods=['POST',"GET"])
    def deploy1(self,kfservice_id):
        mykfservice = db.session.query(KfService).filter_by(id=kfservice_id).first()
        from myapp.utils.py.py_k8s import K8s
        k8s = K8s(mykfservice.project.cluster['KUBECONFIG'])
        namespace = conf.get('KFSERVING_NAMESPACE')
        crd_info = conf.get('CRD_INFO')['inferenceservice']
        crd_list = k8s.get_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                                    namespace=namespace)
        for crd_obj in crd_list:
            if crd_obj['name'] == mykfservice.name:
                k8s.delete_crd(group=crd_info['group'], version=crd_info['version'], plural=crd_info['plural'],
                               namespace=namespace, name=mykfservice.name)
        def get_env(env_str):
            if not env_str:
                return []
            envs = re.split('\r|\n', env_str)
            envs = [env.split('=') for env in envs if env and len(env.split('=')) == 2]
            return envs

        def get_kfjson(service,mykfservice):
            if not service:
                return None

            image_secrets = conf.get('HUBSECRET', [])
            user_hubsecrets = db.session.query(Repository.hubsecret).filter(Repository.created_by_fk == g.user.id).all()
            if user_hubsecrets:
                for hubsecret in user_hubsecrets:
                    if hubsecret[0] not in image_secrets:
                        image_secrets.append(hubsecret[0])

            kfjson={
                "minReplicas": service.min_replicas,
                "maxReplicas": service.max_replicas,
                "custom": {
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "gpu" if core.get_gpu(service.resource_gpu)[0] else "cpu",
                                                "operator": "In",
                                                "values": [
                                                    "true"
                                                ]
                                            },
                                        ]
                                    }
                                ]
                            }
                        },
                    },
                    "imagePullSecrets": [{"name":hubsecret} for hubsecret in image_secrets],
                    "container": {
                        "image": service.images,
                        "imagePullPolicy": 'Always',
                        "name": mykfservice.name+"-"+service.name,
                        "workingDir": service.working_dir if service.working_dir else None,
                        "command": ["sh", "-c",service.command] if service.command else None,
                        "resources": {
                            "requests": {
                                "cpu": service.resource_cpu,
                                "memory": service.resource_memory
                            }
                        },
                        "env":[{"name":env[0],"value":env[1]} for env in get_env(service.env)],
                        # "volumeMounts": [
                        #     {
                        #         "mountPath": "/mnt/%s" % service.created_by.username,
                        #         "name": "workspace",
                        #         "subPath": service.created_by.username
                        #     }
                        # ],
                        # "volumeDevices":[
                        #     {
                        #         "devicePath": "/data/home/",
                        #         "name": "workspace"
                        #     }
                        # ]
                    }
                    # "volumes": [
                    #     {
                    #         "name": "workspace",
                    #         "persistentVolumeClaim": {
                    #             "claimName": "kubeflow-user-workspace"
                    #         }
                    #     }
                    # ]
                }
            }
            return kfjson

        crd_json={
            "apiVersion": "serving.kubeflow.org/v1alpha2",
            "kind": "InferenceService",
            "metadata": {
                "labels": {
                    "app": mykfservice.name
                },
                "name": mykfservice.name,
                "namespace": namespace
            },
            "spec": {
                "canaryTrafficPercent": mykfservice.canary_traffic_percent,
                "default": {
                    mykfservice.service_type: get_kfjson(mykfservice.default_service,mykfservice)
                },
                "canary": {
                    mykfservice.service_type: get_kfjson(mykfservice.canary_service,mykfservice),
                } if mykfservice.canary_service else None,

            }
        }

        import yaml
        ya = yaml.load(json.dumps(crd_json))
        ya_str = yaml.safe_dump(ya, default_flow_style=False)
        logging.info(ya_str)
        crd_objects = k8s.create_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=namespace,body=crd_json)
        flash(category='warning',message='部署启动，一分钟后部署完成')
        return redirect('/kfservice_modelview/list/')

    # 创建kfserving
    @expose('/deploy/<kfservice_id>', methods=['POST', "GET"])
    def deploy(self, kfservice_id):
        mykfservice = db.session.query(KfService).filter_by(id=kfservice_id).first()

        namespace = conf.get('KFSERVING_NAMESPACE')
        crd_info = conf.get('CRD_INFO')['inferenceservice']

        # 根据service生成container
        def make_container(service,mykfservice):
            from myapp.utils.py.py_k8s import K8s
            k8s = K8s()    # 不部署，不需要配置集群信息
            container = k8s.make_container(name=mykfservice.name + "-" + service.name,
                                           command=["sh", "-c",service.command] if service.command else None,
                                           args=None,
                                           volume_mount=None,
                                           image_pull_policy='Always',
                                           image=service.images,
                                           working_dir=service.working_dir if service.working_dir else None,
                                           env=service.env,
                                           resource_memory=service.resource_memory,
                                           resource_cpu = service.resource_cpu,
                                           resource_gpu= service.resource_gpu,
                                           username = service.created_by.username
                                           )
            return container


        api_version = crd_info['group'] + '/' + crd_info['version']
        default_endpoint_spec = V1alpha2EndpointSpec(
            predictor=V1alpha2PredictorSpec(
                min_replicas= mykfservice.default_service.min_replicas,
                max_replicas=mykfservice.default_service.max_replicas,
                custom=V1alpha2CustomSpec(
                    container=make_container(mykfservice.default_service,mykfservice)
                )
            )
        ) if mykfservice.default_service else None

        canary_endpoint_spec = V1alpha2EndpointSpec(
            predictor= V1alpha2PredictorSpec(
                min_replicas=mykfservice.canary_service.min_replicas,
                max_replicas=mykfservice.canary_service.max_replicas,
                custom=V1alpha2CustomSpec(
                    container=make_container(mykfservice.canary_service,mykfservice)
                )
            )
        ) if mykfservice.canary_service else None

        metadata = kubernetes.client.V1ObjectMeta(
            name=mykfservice.name,
            labels={
                "app":mykfservice.name,
                "rtx-user":mykfservice.created_by.username
            },
            namespace=namespace
        )

        isvc = V1alpha2InferenceService(
            api_version=api_version,
            kind=crd_info['kind'],
            metadata=metadata,
            spec=V1alpha2InferenceServiceSpec(
                default=default_endpoint_spec,
                canary=canary_endpoint_spec,
                canary_traffic_percent=mykfservice.canary_traffic_percent
            )
          )

        KFServing = KFServingClient()
        try:
            KFServing.delete(mykfservice.name, namespace=namespace,version=crd_info['version'])
        except Exception as e:
            print(e)

        KFServing.create(isvc,namespace=namespace,version=crd_info['version'])

        flash(category='warning', message='部署启动，一分钟后部署完成')
        return redirect('/kfservice_modelview/list/')


    # 灰度
    @expose('/roll/<kfservice_id>', methods=['POST', "GET"])
    def roll(self, kfservice_id):
        mykfservice = db.session.query(KfService).filter_by(id=kfservice_id).first()
        namespace = conf.get('KFSERVING_NAMESPACE')
        crd_info = conf.get('CRD_INFO')['inferenceservice']

        # 根据service生成container
        def make_container(service, mykfservice):
            from myapp.utils.py.py_k8s import K8s
            k8s = K8s()   # 不部署，不需要配置集群信息
            container = k8s.make_container(name=mykfservice.name + "-" + service.name,
                                           command=["sh", "-c", service.command] if service.command else None,
                                           args=None,
                                           volume_mount=None,
                                           image_pull_policy='Always',
                                           image=service.images,
                                           working_dir=service.working_dir if service.working_dir else None,
                                           env=service.env,
                                           resource_memory=service.resource_memory,
                                           resource_cpu=service.resource_cpu,
                                           resource_gpu=service.resource_gpu,
                                           username=service.created_by.username,
                                           ports = service.ports
                                           )
            return container


        canary_endpoint_spec = V1alpha2EndpointSpec(
            predictor=V1alpha2PredictorSpec(
                min_replicas=mykfservice.canary_service.min_replicas,
                max_replicas=mykfservice.canary_service.max_replicas,
                custom=V1alpha2CustomSpec(
                    container=make_container(mykfservice.canary_service, mykfservice)
                )
            )
        ) if mykfservice.canary_service else None

        KFServing = KFServingClient()
        KFServing.rollout_canary(mykfservice.name, canary=canary_endpoint_spec, percent=mykfservice.canary_traffic_percent,
                                 namespace=namespace, timeout_seconds=120,version=crd_info['version'])

        flash(category='warning', message='滚动升级已配置，刷新查看当前流量比例')
        return redirect('/kfservice_modelview/list/')

    # 基础批量删除
    # @pysnooper.snoop()
    def base_muldelete(self,items):
        if not items:
            abort(404)
        for item in items:
            try:
                k8s_client = py_k8s.K8s(item.project.cluster['KUBECONFIG'])
                crd_info = conf.get("CRD_INFO", {}).get(self.crd_name, {})
                if crd_info:
                    k8s_client.delete_crd(group=crd_info['group'],version=crd_info['version'],plural=crd_info['plural'],namespace=conf.get('KFSERVING_NAMESPACE'),name=item.name)
            except Exception as e:
                flash(str(e), "danger")

    def pre_delete(self,item):
        self.base_muldelete([item])

    # @event_logger.log_this
    # @expose("/delete/<pk>")
    # @has_access
    # def delete(self, pk):
    #     pk = self._deserialize_pk_if_composite(pk)
    #     self.base_delete(pk)
    #     url = url_for(f"{self.endpoint}.list")
    #     return redirect(url)

appbuilder.add_view(KfService_ModelView,"kfserving",icon = 'fa-tasks',category = '服务化')




