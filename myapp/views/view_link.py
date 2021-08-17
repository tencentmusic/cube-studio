
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
# 将model添加成视图，并控制在前端的显示
from myapp import app, appbuilder,db,event_logger

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
from types import FunctionType
from flask_appbuilder import CompactCRUDMixin, expose
import pysnooper,datetime,time,json
conf = app.config
from myapp.views.base import BaseMyappView


all_links = conf.get('ALL_LINKS',{})
for link in all_links:
    appbuilder.add_link(
        link['label'],
        label=_(link['label']),
        href=link['url'],
        category_icon="fa-link",
        icon="fa-link",
        category="link",
        category_label=__("链接"),
    )



