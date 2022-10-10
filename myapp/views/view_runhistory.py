from flask import render_template,redirect
from flask_appbuilder.models.sqla.interface import SQLAInterface


from myapp.models.model_job import RunHistory

from myapp import app, appbuilder,db,event_logger
from wtforms import SelectField
from flask_appbuilder.fieldwidgets import Select2Widget
from flask_babel import lazy_gettext as _
from sqlalchemy import and_, or_, select

from .baseApi import (
    MyappModelRestApi
)

from myapp import security_manager
from .base import (
    DeleteMixin,
    MyappFilter,
    MyappModelView,
)
conf = app.config
logging = app.logger


class RunHistory_Filter(MyappFilter):
    # @pysnooper.snoop()
    def apply(self, query, func):
        user_roles = [role.name.lower() for role in list(self.get_user_roles())]
        if "admin" in user_roles:
            return query

        pipeline_ids = security_manager.get_create_pipeline_ids(db.session)
        return query.filter(
            or_(
                self.model.pipeline_id.in_(pipeline_ids),
                # self.model.project.name.in_(['public'])
            )
        )



class RunHistory_ModelView_Base():
    label_title='定时调度历史'
    datamodel = SQLAInterface(RunHistory)
    base_order = ('id', 'desc')
    order_columns = ['id']
    base_permissions = ['can_show', 'can_list', 'can_delete']
    list_columns = ['pipeline_url','creator','created_on','execution_date','status_url','log','history']
    cols_width={
        "pipeline_url": {"type": "ellip2", "width": 400},
        "created_on":{"type": "ellip2", "width": 300}
    }
    edit_columns = ['status']
    base_filters = [["id", RunHistory_Filter, lambda: []]]
    add_form_extra_fields = {
        "status": SelectField(
            _(datamodel.obj.lab('status')),
            description="状态comed为已识别未提交，created为已提交",
            widget=Select2Widget(),
            choices=[['comed', 'comed'], ['created', 'created']]
        ),
    }
    edit_form_extra_fields = add_form_extra_fields


class RunHistory_ModelView(RunHistory_ModelView_Base,MyappModelView,DeleteMixin):
    datamodel = SQLAInterface(RunHistory)

appbuilder.add_view_no_menu(RunHistory_ModelView)




# 添加api
class RunHistory_ModelView_Api(RunHistory_ModelView_Base,MyappModelRestApi):
    datamodel = SQLAInterface(RunHistory)
    route_base = '/runhistory_modelview/api'

appbuilder.add_api(RunHistory_ModelView_Api)



