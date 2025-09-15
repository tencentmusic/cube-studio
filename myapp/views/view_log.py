from flask_appbuilder import action

from myapp.views.baseSQLA import MyappSQLAInterface
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder
from myapp.models.log import Log
from myapp.views.baseApi import MyappModelRestApi

class LOG_ModelView_Api(MyappModelRestApi):
    label_title = _('日志')
    route_base = '/log_modelview/api'
    datamodel = MyappSQLAInterface(Log)
    list_columns = ['user','method','path','duration_ms','dttm']
    base_permissions = ['can_list']
    search_columns = ["user",'method','path']
    base_order = ("dttm", "desc")
    order_columns = ['id']
    spec_label_columns = {
        "action": _("函数"),
        "path": _("网址"),
        "dttm": _("时间"),
        "duration_ms": _("响应延迟"),
        "referrer": _("相关人"),
    }

    @action("muldelete", "删除", "确定删除所选记录?", "fa-trash", single=False)
    def muldelete(self, items):
        return self._muldelete(items)

appbuilder.add_api(LOG_ModelView_Api)




