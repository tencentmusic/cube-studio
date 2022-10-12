
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _

from myapp import app, appbuilder

conf = app.config


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



