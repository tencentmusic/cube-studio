from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_babel import gettext as __

from myapp import app, appbuilder
from myapp.models.log import Log
from myapp.views.base import MyappModelView
from . import LogMixin


class LogModelView(LogMixin, MyappModelView):
    datamodel = SQLAInterface(Log)
    list_columns = ['user','method','path','duration_ms','dttm']


if (
    not app.config.get("FAB_ADD_SECURITY_VIEWS") is False
    or app.config.get("MYAPP_LOG_VIEW") is False
):
    appbuilder.add_view(
        LogModelView,
        "Action Log",
        label=__("Action Log"),
        category="Security",
        category_label=__("Security"),
        icon="fa-list-ol",
    )





