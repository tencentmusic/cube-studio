
from flask_appbuilder import ModelRestApi
from myapp.views.baseSQLA import MyappSQLAInterface
from myapp import app, appbuilder
from myapp.models.log import Log
from . import LogMixin


class LogRestApi(LogMixin, ModelRestApi):
    datamodel = MyappSQLAInterface(Log)


    class_permission_name = "LogModelView"
    method_permission_name = {
        "get_list": "list",
        "get": "show",
        "post": "add",
        "put": "edit",
        "delete": "delete",
        "info": "list",
    }
    resource_name = "log"
    allow_browser_login = True
    list_columns = ("user.username", "action", "dttm")


if (
    not app.config.get("FAB_ADD_SECURITY_VIEWS") is False
    or app.config.get("MYAPP_LOG_VIEW") is False
):
    appbuilder.add_api(LogRestApi)
