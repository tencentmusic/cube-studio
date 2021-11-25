
from flask_babel import lazy_gettext as _


class LogMixin:
    label_title='日志'

    list_columns = ("user", "action", "dttm")
    edit_columns = ("user", "action", "dttm", "json")
    base_order = ("dttm", "desc")
    # label_columns = {
    #     "user": _("User"),
    #     "action": _("Action"),
    #     "dttm": _("dttm"),
    #     "json": _("JSON"),
    # }
