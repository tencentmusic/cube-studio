import datetime
import json
import random
import time

from cubestudio.request.model import Model
from cubestudio.request.model_client import client,init
from cubestudio.request.model import Job_Template,Project,Pipeline,Task,Images

if __name__=="__main__":
    HOST = "http://host.docker.internal:80"
    token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJwZW5nbHVhbiJ9.3WrxmDOK7PMt0P2xdgUx-1HLvhgeNRPKFaeQzFiIkoU'
    init(host=HOST,username='pengluan',token=token)
    job_template = client(Job_Template).one(name="自定义镜像")
    print(job_template)
    # 添加一个画布
    pipeline = client(Pipeline).add_or_update(
        name='pengluan-default',
        describe='sdk画布',
        project=client(Project).one(name='public')
    )
    print(pipeline)
    task=client(Task).add_or_update(
        name='sdk-test1',
        label='sdk发起的任务',
        pipeline=pipeline,
        job_template=client(Job_Template).one(name="自定义镜像"),
        args=json.dumps(
            {
                "images":"ubuntu:18.04",
                "command":'for i in {1..100}; do date; sleep 1; done',
                "workdir":"/"
            }
        )
    )
    print(task)
    task.run()
    task.log(follow=True)
    task.stop()



