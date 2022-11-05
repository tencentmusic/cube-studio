mkidr -p /data/log/nginx/

nginx -g "daemon off;" &

# celery --app=cubestudio.aihub.web.celery_app:celery_app worker -Q app1 --loglevel=info --pool=prefork -Ofair -c 10

python app.py
#exec "$@"
