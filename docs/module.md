# 各pod功能
平台控制端为fab框架，可以参考https://github.com/tencentmusic/fab

 - deploy.yaml为前后端服务的容器
 - deploy-schedule.yaml为产生平台celery任务的pod(非pipeline任务)
 - deploy-worker.yaml为执行平台celery任务的pod(非pipeline任务)
 - deploy-watch.yaml为监听容器，会监听平台中的crd的状态变化，进而更新数据库和推送消息

