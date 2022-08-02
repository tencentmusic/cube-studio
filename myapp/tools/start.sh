

nohup python watch_workflow.py > workflow.log 2>&1 &

nohup python watch_service.py > service.log 2>&1 &

tail -f workflow.log service.log
