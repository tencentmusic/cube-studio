
"""Utility functions used across Myapp"""
import os.path
import logging
import pysnooper
import datetime
import time
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp.utils.py.py_k8s import K8s
from myapp.utils.celery import session_scope
from myapp.project import push_message,push_admin
from myapp.tasks.celery_app import celery_app
import importlib
from myapp import app
from myapp.models.model_serving import InferenceService
from myapp.models.model_dataset import Dataset
from myapp.views.view_inferenceserving import InferenceService_ModelView_base
from myapp.models.model_docker import Docker
from myapp.models.model_notebook import Notebook
conf = app.config


@celery_app.task(name="task.check_docker_commit", bind=True)  # , soft_time_limit=15
def check_docker_commit(task,docker_id):  # 在页面中测试时会自定接收者和id
    logging.info('============= begin run check_docker_commit task')
    with session_scope(nullpool=True) as dbsession:
        try:
            docker = dbsession.query(Docker).filter_by(id=int(docker_id)).first()
            pod_name = "docker-commit-%s-%s" % (docker.created_by.username, str(docker.id))
            namespace = conf.get('NOTEBOOK_NAMESPACE')
            k8s_client = K8s(conf.get('CLUSTERS').get(conf.get('ENVIRONMENT')).get('KUBECONFIG',''))
            begin_time=datetime.datetime.now()
            now_time=datetime.datetime.now()
            while((now_time-begin_time).total_seconds()<1800):   # 也就是最多commit push 30分钟
                time.sleep(60)
                commit_pods = k8s_client.get_pods(namespace=namespace,pod_name=pod_name)
                if commit_pods:
                    commit_pod=commit_pods[0]
                    if commit_pod['status']=='Succeeded':
                        docker.last_image=docker.target_image
                        dbsession.commit()
                        break
                    # 其他异常状态直接报警
                    if commit_pod['status']!='Running':
                        push_message(conf.get('ADMIN_USER').split(','),'commit pod %s not running'%commit_pod['name'])
                        break
                else:
                    break

        except Exception as e:
            logging.error(e)


@celery_app.task(name="task.check_notebook_commit", bind=True)  # , soft_time_limit=15
def check_notebook_commit(task,notebook_id,target_image):  # 在页面中测试时会自定接收者和id
    logging.info('============= begin run check_notebook_commit task')
    with session_scope(nullpool=True) as dbsession:
        try:
            notebook = dbsession.query(Notebook).filter_by(id=int(notebook_id)).first()
            pod_name = "notebook-commit-%s-%s" % (notebook.created_by.username, str(notebook.id))
            namespace = notebook.namespace
            k8s_client = K8s(notebook.cluster.get('KUBECONFIG', ''))
            begin_time=datetime.datetime.now()
            now_time=datetime.datetime.now()
            while((now_time-begin_time).total_seconds()<1800):   # 也就是最多commit push 30分钟
                time.sleep(60)
                commit_pods = k8s_client.get_pods(namespace=namespace,pod_name=pod_name)
                if commit_pods:
                    commit_pod=commit_pods[0]
                    if commit_pod['status']=='Succeeded':
                        notebook.images=target_image
                        dbsession.commit()
                        push_message([notebook.created_by.username],'notebook %s save success'%notebook.name)
                        break
                    # 其他异常状态直接报警
                    if commit_pod['status']!='Running':
                        push_message([notebook.created_by.username], 'notebook %s save fail' % notebook.name)
                        break
                else:
                    break

        except Exception as e:
            logging.error(e)

@celery_app.task(name="task.upgrade_service", bind=True)  # , soft_time_limit=15
def upgrade_service(task,service_id,name,namespace):
    logging.info('============= begin run upgrade_service task')
    # 将旧的在线版本进行下掉，前提是新的服务必须已经就绪
    time.sleep(10)
    with session_scope(nullpool=True) as dbsession:
        try:
            service = dbsession.query(InferenceService).filter_by(id=int(service_id)).first()
            message = __('%s 准备进行服务迭代 %s %s')%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),service.model_name,service.model_version)
            # push_admin(message)
            push_message([service.created_by.username],message)
            k8s_client = K8s(service.project.cluster.get('KUBECONFIG',''))
            begin_time = time.time()
            crd_info = conf.get("CRD_INFO", {}).get('virtualservice', {})

            # 按单台机器滚动升级
            # 先删除老服务的hpa
            # @pysnooper.snoop()
            def set_wight(service,old_service,new_serviec_weight):
                # 修改原来的vs的权重，新的不动
                weight_body = {
                    "spec": {
                        "http": [
                            {
                                "route": [
                                    {
                                        "destination": {
                                            "host": f"{service.name}.service.svc.cluster.local",
                                            "port": {
                                                "number": int(
                                                    service.ports.split(',')[0])
                                            }
                                        },
                                        "weight": new_serviec_weight
                                    },
                                    {
                                        "destination": {
                                            "host": f"{old_service.name}.service.svc.cluster.local",
                                            "port": {
                                                "number": int(
                                                    old_service.ports.split(',')[0])
                                            }
                                        },
                                        "weight": 100 - new_serviec_weight
                                    }
                                ],
                                "timeout": "3000s",
                            }
                        ]
                    }
                }
                crd_object = k8s_client.CustomObjectsApi.get_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    plural=crd_info['plural'],
                    namespace=namespace,
                    name=old_service.name
                )
                crd_object['spec']['http'][0]=weight_body['spec']['http'][0]

                # logging.info(crd)
                # logging.info(crd_object)
                old_virtual_service = k8s_client.CustomObjectsApi.replace_namespaced_custom_object(
                    group=crd_info['group'],
                    version=crd_info['version'],
                    plural=crd_info['plural'],
                    namespace=namespace,
                    name=old_service.name,
                    body=crd_object
                )


            while (True):
                try:
                    deployment = k8s_client.AppsV1Api.read_namespaced_deployment(name=name, namespace=namespace)
                    if deployment:
                        ready_replicas = deployment.status.ready_replicas
                        replicas = deployment.status.replicas
                        message = __('%s 服务 %s %s ready副本数:%s 目标副本数:%s') % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), service.model_name,service.model_version, ready_replicas, replicas)
                        # push_admin(message)
                        push_message([service.created_by.username], message)
                        # 如果新的dp副本数已全部就绪
                        if ready_replicas == replicas:
                            break
                        else:
                            old_service=None
                            # 没有配置域名也就不用切换流量
                            if not service.host:
                                # 考虑资源不足,新服务部署不起来，此时需要缩放旧服务
                                old_service = dbsession.query(InferenceService) \
                                    .filter(InferenceService.model_status == 'online') \
                                    .filter(InferenceService.model_name == service.model_name) \
                                    .filter(InferenceService.name != service.name) \
                                    .filter(InferenceService.host == service.host).first()
                            # 有旧服务才改流量
                            if old_service:
                                old_deployment = k8s_client.AppsV1Api.read_namespaced_deployment(name=old_service.name,namespace=namespace)
                                # 先更改流量比例
                                crd = k8s_client.get_one_crd(
                                    group=crd_info['group'],
                                    version=crd_info['version'],
                                    plural=crd_info['plural'],
                                    namespace=namespace,
                                    name=old_service.name
                                )
                                # 一定是有新pod才更改流量比例
                                if old_deployment and crd and deployment.status.ready_replicas>0:

                                    # 按当前情况修改旧服务的vs流量比例，新服务的vs 因为重名不生效
                                    new_serviec_weight = deployment.status.ready_replicas*100//(deployment.status.ready_replicas+old_deployment.status.ready_replicas)
                                    # 如果旧服务只有1台了，那就直接100%到新服务上
                                    if old_deployment.status.ready_replicas==1:
                                        new_serviec_weight=100
                                    # 修改原来的vs的权重，新的不动
                                    set_wight(service,old_service, new_serviec_weight)
                                # time.sleep(10)
                                # 再缩放副本
                                if old_deployment and old_deployment.status.ready_replicas>0:
                                    scale_body = {
                                        "spec": {
                                            "replicas": old_deployment.status.ready_replicas-1
                                        }
                                    }

                                    api_response = k8s_client.AppsV1Api.patch_namespaced_deployment_scale(
                                        name=old_service.name,
                                        namespace=namespace,
                                        body=scale_body
                                    )

                    else:
                        message = __('%s 没有发现 新服务 %s %s 的 deployment，部署失败') % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), service.model_name,service.model_version)
                        # push_admin(message)
                        push_message([service.created_by.username], message)
                        return

                except Exception as e:
                    logging.error(e)

                if time.time() - begin_time > 600:
                    message = __('%s 新版本运行状态检查超时，请手动检查和清理旧版本%s %s') % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), service.model_name, service.model_version)
                    # push_admin(message)
                    push_message([service.created_by.username], message)
                    return
                time.sleep(60)


            # 切换还完，做旧服务的清理。 同域名的只能保留一个，这样能让客户端使用同一个域名总是请求到最新的一个服务。但是要避免service.host配置的不是域名的情况
            def get_inference_host(inference):

                service_host = inference.name + "." +inference.project.cluster.get('SERVICE_DOMAIN', conf.get('SERVICE_DOMAIN', ''))

                if service.host:
                    from myapp.utils.core import split_url
                    host, port, path = split_url(inference.host)
                    if host:
                        service_host = host

                return service_host

            old_services = dbsession.query(InferenceService)\
                .filter(InferenceService.model_status=='online')\
                .filter(InferenceService.model_name==service.model_name)\
                .filter(InferenceService.name!=service.name).all()
            old_services = [service1 for service1 in old_services if get_inference_host(service1)==get_inference_host(service)]

            if old_services:
                for old_service in old_services:
                    if old_service.name != service.name:
                        inference_model_view = InferenceService_ModelView_base()
                        inference_model_view.delete_old_service(old_service.name, old_service.project.cluster)
                        old_service.model_status = 'offline'
                        old_service.deploy_history = service.deploy_history + "\n" + "clear: %s %s" % ('admin', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        dbsession.commit()
                        message = __('%s 新版本服务升级完成，下线旧服务 %s %s') % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),service.model_name, old_service.model_version)
                        # push_admin(message)
                        push_message([service.created_by.username],message)
            else:
                message = __('%s %s 没有历史在线版本，%s版本升级完成') % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), service.model_name, service.model_version)
                # push_admin(message)
                push_message([service.created_by.username], message)

        except Exception as e:
            logging.error(e)
            push_admin(__('部署升级报错 %s %s: %s') % (service.model_name, service.model_version,str(e)))


@celery_app.task(name="task.update_dataset", bind=True)  # , soft_time_limit=15
# @pysnooper.snoop()
def update_dataset(task,dataset_id):
    logging.info('============= begin run update_dataset task')
    with session_scope(nullpool=True) as dbsession:
        try:
            dataset = dbsession.query(Dataset).filter_by(id=dataset_id).first()

            remote_dir = f'dataset/{dataset.name}/{dataset.version if dataset.version else "latest"}/'
            remote_dir = os.path.join('/data/k8s/kubeflow/global/', remote_dir)
            if os.path.exists(remote_dir):
                # 先清理干净，因为有可能存在旧的不对的数据
                import shutil
                shutil.rmtree(remote_dir, ignore_errors=True)
            os.makedirs(remote_dir, exist_ok=True)

            # 备份在本地
            if dataset.path:
                paths = dataset.path.split("\n")
                for path in paths:
                    file_name = path[path.rindex("/") + 1:]
                    local_path = os.path.join('/home/myapp/myapp/static/', path.lstrip('/'))
                    if os.path.exists(local_path):
                        # 对文件直接复制
                        if os.path.isfile(local_path):
                            shutil.copy(local_path,remote_dir)
                        # 对文件夹要拷贝文件夹
                        if os.path.isdir(local_path):
                            shutil.copytree(local_path,remote_dir)

            elif dataset.download_url:
                download_urls = dataset.download_url.split("\n")
                for download_url in download_urls:
                    try:
                        import requests
                        filename = download_url.split("/")[-1]
                        try_num=0
                        while try_num<3:
                            try_num+=1
                            response = requests.get(download_url)
                            with open(remote_dir + '/' + filename, 'wb') as f:
                                f.write(response.content)
                                break
                    except Exception as e:
                        print(e)

        except Exception as e:
            logging.error(e)
            push_admin(f'数据集备份失败，id:{dataset_id}')


if __name__ =='__main__':
    upgrade_service(task=None,service_id=21,namespace='service',name='serving-nginx-202303141')
