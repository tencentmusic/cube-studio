import requests,os
from celery import Celery
from celery.result import AsyncResult
redis_url_default = 'redis://:admin@127.0.0.1:6379/0'
CELERY_BROKER_URL = os.getenv('REDIS_URL', redis_url_default)
CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', redis_url_default)

celery_app = Celery(broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# 定义认为放在的队列
@celery_app.task(queue=os.getenv('APPNAME','app1'))
def inference(name, version, data):
    try:
        res = requests.post(f'http://127.0.0.1:8080/{name}/api/model/{name}/version/{version}/',json=data)
        result = res.json()
        return result
    except Exception as e:
        print(e)
        return {
            "status": 1,
            "result": [],
            "message": "推理失败了"
        }
