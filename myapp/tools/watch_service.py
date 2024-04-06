import logging
logging.basicConfig(format='%(asctime)s:watch-service:%(levelname)s:%(message)s', level=logging.INFO,datefmt = '%Y-%m-%d %H:%M:%S')
import time, os
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from kubernetes import client
from kubernetes import watch
from myapp.utils.py.py_k8s import K8s
from myapp.project import push_message
from myapp import app
from myapp.utils.celery import session_scope
conf = app.config

cluster = os.getenv('ENVIRONMENT', '').lower()
if not cluster:
    logging.info('no cluster %s' % cluster)
    exit(1)
else:
    clusters = conf.get('CLUSTERS', {})
    if clusters and cluster in clusters:
        kubeconfig = clusters[cluster].get('KUBECONFIG', '')
        K8s(kubeconfig)
    else:
        logging.error('no kubeconfig in cluster %s' % cluster)
        exit(1)

from myapp.models.model_serving import InferenceService
from datetime import datetime, timezone, timedelta


# @pysnooper.snoop()
def listen_service():
    namespace = conf.get('SERVICE_NAMESPACE')
    w = watch.Watch()
    while (True):
        try:
            logging.info('begin listen')
            for event in w.stream(client.CoreV1Api().list_namespaced_pod, namespace=namespace,timeout_seconds=60):  # label_selector=label,
                with session_scope(nullpool=True) as dbsession:
                    try:
                        if event['object'].status and event['object'].status.container_statuses and event["type"]=='MODIFIED':  # 容器重启会触发MODIFIED
                            # terminated 终止，waiting 等待启动，running 运行中
                            container_statuse = event['object'].status.container_statuses[0].state
                            terminated = container_statuse.terminated
                            # waiting = container_statuse.waiting
                            # running = container_statuse.running
                            service_name=event['object'].metadata.labels.get('app','')
                            inferenceserving = dbsession.query(InferenceService).filter_by(name=service_name).first() if service_name else None
                            if service_name and inferenceserving:
                                # print(event['object'].status)
                                if terminated and terminated.finished_at:  # 任务终止
                                    finished_at = int(terminated.finished_at.astimezone(timezone(timedelta(hours=8))).timestamp())  # 要找事件发生的时间
                                    if (datetime.now().timestamp() - finished_at) < 5:
                                        message = "cluster: %s, pod: %s, user: %s, status: %s" % (cluster,event['object'].metadata.name,inferenceserving.created_by.username, 'terminated')
                                        logging.info(message)
                                        push_message([inferenceserving.created_by.username], message)
                                # if running and running.started_at:  # 任务重启运行
                                #     start_time = int(running.started_at.astimezone(timezone(timedelta(hours=8))).timestamp())  # 要找事件发生的时间
                                #     if (datetime.now().timestamp() - start_time) < 5:
                                #         message = "pod %s %s" % (event['object'].metadata.name, 'running')
                                #         push_message([inferenceserving.created_by.username]+conf.get('ADMIN_USER').split(','), message)

                    except Exception as e:
                        logging.error(e)

        except Exception as ee:
            logging.error(ee)
            time.sleep(5)


# 不能使用异步io，因为stream会阻塞
if __name__ == '__main__':
    listen_service()
