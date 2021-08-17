
"""Utility functions used across Myapp"""

# Myapp framework imports
from myapp import app
from myapp.utils.core import get_celery_app

# 全局配置，全部celery app。所有任务都挂在这个app下面
conf = app.config
# print(conf)
# 获取CELERY_CONFIG中定义的任务配置，这个是所有任务的统一配置项，所有任务都应用这个配置项。
# 但这并不是真正的定义任务。因为任务是task修饰定义的。并且是有逻辑的
celery_app = get_celery_app(conf)




