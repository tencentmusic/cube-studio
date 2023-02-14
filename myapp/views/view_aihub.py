
from flask_appbuilder.models.sqla.interface import SQLAInterface
import urllib.parse
from myapp import app, appbuilder,db
from wtforms import SelectField
from flask_appbuilder.fieldwidgets import Select2Widget
from myapp.models.model_job import Images,Job_Template,Repository
from myapp.models.model_team import Project,Project_User
from myapp.models.model_serving import InferenceService


from .baseApi import (
    MyappModelRestApi
)
from flask import (
    flash,
    redirect
)
from .base import (
    MyappFilter,
)
from myapp.models.model_aihub import Aihub
from flask_appbuilder import expose
import datetime,json
conf = app.config
logging = app.logger


def add_job_template_group(name, describe, expand={}):
    project = db.session.query(Project).filter_by(name=name).filter_by(type='job-template').first()
    if project is None:
        try:
            project = Project()
            project.type = 'job-template'
            project.name = name
            project.describe = describe
            project.expand = json.dumps(expand, ensure_ascii=False, indent=4)
            db.session.add(project)
            db.session.commit()

            project_user = Project_User()
            project_user.project = project
            project_user.role = 'creator'
            project_user.user_id = 1
            db.session.add(project_user)
            db.session.commit()
            print('add project %s' % name)
        except Exception as e:
            print(e)
            db.session.rollback()


def create_template(project_name, image_name, image_describe, job_template_name,
                    job_template_old_names=[], job_template_describe='', job_template_command='',
                    job_template_args=None, job_template_volume='', job_template_account='', job_template_expand=None,
                    job_template_env='', gitpath=''):

    repository = db.session.query(Repository).filter_by(name='hubsecret').first()

    images = db.session.query(Images).filter_by(name=image_name).first()
    project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
    # 创建分组
    if not project:
        add_job_template_group(project_name,project_name)
    # 创建镜像
    if images is None and project:
        try:
            images = Images()
            images.name = image_name
            images.describe = image_describe
            images.created_by_fk = 1
            images.changed_by_fk = 1
            images.project_id = project.id
            images.repository_id = repository.id
            images.gitpath = gitpath
            db.session.add(images)
            db.session.commit()
            print('add images %s' % image_name)
        except Exception as e:
            print(e)
            db.session.rollback()
    # 创建模板
    job_template = db.session.query(Job_Template).filter_by(name=job_template_name).first()

    project = db.session.query(Project).filter_by(name=project_name).filter_by(type='job-template').first()
    if project and images.id:
        if job_template is None:
            try:
                job_template = Job_Template()
                job_template.name = job_template_name.replace('_', '-')
                job_template.describe = job_template_describe
                job_template.entrypoint = job_template_command
                job_template.volume_mount = job_template_volume
                job_template.accounts = job_template_account
                job_template_expand['source'] = "aihub"
                job_template.expand = json.dumps(job_template_expand, indent=4,ensure_ascii=False) if job_template_expand else '{}'
                job_template.created_by_fk = 1
                job_template.changed_by_fk = 1
                job_template.project_id = project.id
                job_template.images_id = images.id
                job_template.version = 'Release'
                job_template.env = job_template_env
                job_template.args = json.dumps(job_template_args, indent=4,
                                               ensure_ascii=False) if job_template_args else '{}'
                db.session.add(job_template)
                db.session.commit()
                print('add job_template %s' % job_template_name.replace('_', '-'))
            except Exception as e:
                print(e)
                db.session.rollback()
        else:
            try:
                job_template.name = job_template_name.replace('_', '-')
                job_template.describe = job_template_describe
                job_template.entrypoint = job_template_command
                job_template.volume_mount = job_template_volume
                job_template.accounts = job_template_account
                job_template_expand['source'] = "github"
                job_template.expand = json.dumps(job_template_expand, indent=4,
                                                 ensure_ascii=False) if job_template_expand else '{}'
                job_template.created_by_fk = 1
                job_template.changed_by_fk = 1
                job_template.project_id = project.id
                job_template.images_id = images.id
                job_template.version = 'Release'
                job_template.env = job_template_env
                job_template.args = json.dumps(job_template_args, indent=4,
                                               ensure_ascii=False) if job_template_args else '{}'
                db.session.commit()
                print('update job_template %s' % job_template_name.replace('_', '-'))
            except Exception as e:
                print(e)
                db.session.rollback()



# 添加 demo 推理 服务
# @pysnooper.snoop()
def create_inference(project_name,service_name,service_describe,image_name,command,env,model_name,workdir='',model_version='',model_path='',service_type='serving',resource_memory='2G',resource_cpu='2',resource_gpu='0',ports='80',volume_mount='kubeflow-user-workspace(pvc):/mnt',metrics='',health='',inference_config='',expand={}):
    service = db.session.query(InferenceService).filter_by(name=service_name).first()
    project = db.session.query(Project).filter_by(name=project_name).filter_by(type='org').first()
    if service is None and project:
        try:
            service = InferenceService()
            service.name = service_name.replace('_','-')
            service.label=service_describe
            service.service_type=service_type
            service.model_name=model_name
            service.model_version=model_version if model_version else datetime.now().strftime('v%Y.%m.%d.1')
            service.model_path = model_path
            service.created_by_fk=1
            service.changed_by_fk=1
            service.project_id=project.id
            service.project=project
            service.images=image_name
            service.resource_memory=resource_memory
            service.resource_cpu=resource_cpu
            service.resource_gpu = resource_gpu
            service.working_dir=workdir
            service.command = command
            service.inference_config = inference_config
            service.env='\n'.join([x.strip() for x in env.split('\n') if x.split()])
            service.ports = ports
            service.volume_mount=volume_mount
            service.metrics=metrics
            service.health=health
            service.expand = json.dumps(expand,indent=4,ensure_ascii=False)

            from myapp.views.view_inferenceserving import InferenceService_ModelView_base
            inference_class = InferenceService_ModelView_base()
            inference_class.src_item_json = {}
            inference_class.pre_add(service)

            db.session.add(service)
            db.session.commit()
            print('add inference %s' % service_name)
        except Exception as e:
            print(e)
            db.session.rollback()


# 获取某类project分组
class Aihub_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, value):
        # user_roles = [role.name.lower() for role in list(get_user_roles())]
        # if "admin" in user_roles:
        #     return query.filter(Project.type == value).order_by(Project.id.desc())
        return query.filter(self.model.field==value).order_by(self.model.id.desc())


class Aihub_base():
    label_title='模型市场'
    datamodel = SQLAInterface(Aihub)
    base_permissions = ['can_show','can_list']
    base_order = ("hot", "desc")
    order_columns = ['id']
    search_columns=['describe','label','name','field','scenes']
    list_columns = ['card']

    spec_label_columns={
        "name":"英文名",
        "field": "领域",
        "label": "中文名",
        "describe":"描述",
        "scenes":"场景",
        "card": "信息"
    }

    edit_form_extra_fields = {
        "field": SelectField(
            label='AI领域',
            description='AI领域',
            widget=Select2Widget(),
            default='',
            choices=[['机器视觉','机器视觉'], ['听觉','听觉'],['自然语言', '自然语言'],['强化学习', '强化学习'],['图论', '图论'], ['通用','通用']]
        ),
    }


    def post_list(self,items):
        flash('AIHub内容同步于github，<a target="_blank" href="https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning">参与贡献</a>',category='success')
        return items


    # @event_logger.log_this
    @expose('/notebook/<aihub_id>',methods=['GET','POST'])
    def notebook(self,aihub_id):
        aihub = db.session.query(Aihub).filter_by(uuid=aihub_id).first()
        try:
            if aihub and aihub.notebook:
                notebook = json.loads(aihub.notebook)
                return redirect(notebook.get("jupyter",[])[0])
        except Exception as e:
            print(e)
        return redirect(aihub.doc)


    # @event_logger.log_this
    @expose('/train/<aihub_id>',methods=['GET','POST'])
    def train(self,aihub_id):
        aihub = db.session.query(Aihub).filter_by(uuid=aihub_id).first()
        try:
            if aihub and aihub.job_template:
                job_template = json.loads(aihub.job_template)
                create_template(**job_template)
                flash('任务模板已注册，拖拉模板配置训练任务','success')
                url = conf.get('MODEL_URLS', {}).get('job_template', '') + '?filter=' + urllib.parse.quote(
                    json.dumps([{"key": "name", "value": job_template.get('job_template_name','')}], ensure_ascii=False))
                print(url)
                return redirect(url)

        except Exception as e:
            print(e)
        return redirect(aihub.doc)


    # @event_logger.log_this
    @expose('/service/<aihub_id>',methods=['GET','POST'])
    def service(self,aihub_id):
        aihub = db.session.query(Aihub).filter_by(uuid=aihub_id).first()
        try:
            if aihub and aihub.inference:
                inference = json.loads(aihub.inference)
                create_inference(**inference)
                flash('服务已注册，部署后访问','success')
                url = conf.get('MODEL_URLS', {}).get('inferenceservice', '') + '?filter=' + urllib.parse.quote(
                    json.dumps([{"key": "name", "value": inference.get('service_name', '')}],
                               ensure_ascii=False))
                print(url)
                return redirect(url)
        except Exception as e:
            print(e)
        return redirect(aihub.doc)



class Aihub_visual_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/visual/api'
    base_filters = [["id", Aihub_Filter, 'visual']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_visual_Api)


class Aihub_voice_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/voice/api'
    base_filters = [["id", Aihub_Filter, 'voice']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_voice_Api)


class Aihub_language_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/language/api'
    base_filters = [["id", Aihub_Filter, 'language']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_language_Api)


class Aihub_reinforcement_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/reinforcement/api'
    base_filters = [["id", Aihub_Filter, 'reinforcement']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_reinforcement_Api)

class Aihub_graph_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/graph/api'
    base_filters = [["id", Aihub_Filter, 'graph']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_graph_Api)

class Aihub_common_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/common/api'
    base_filters = [["id", Aihub_Filter, 'common']]
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['isCard']=True

appbuilder.add_api(Aihub_common_Api)



class Aihub_Api(Aihub_base,MyappModelRestApi):
    route_base = '/aihub/api'
    # @pysnooper.snoop()
    def add_more_info(self,response,**kwargs):
        response['list_ui_type']='card'
        response['list_ui_args']={
            "card_width":'385px',
            "card_heigh": '250px'
        }


appbuilder.add_api(Aihub_Api)



