

nohup python myapp/tools/watch_workflow.py > workflow.log 2>&1 &

nohup python myapp/tools/watch_service.py > service.log 2>&1 &

tail -f workflow.log service.log
