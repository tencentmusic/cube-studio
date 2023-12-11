from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
class LogMixin:
    label_title= _('日志')

    list_columns = ("user", "action", "dttm")
    edit_columns = ("user", "action", "dttm", "json")
    base_order = ("dttm", "desc")
    # label_columns = {
    #     "user": "User",
    #     "action": "Action",
    #     "dttm": "dttm",
    #     "json": "JSON",
    # }
