mkidr -p /data/log/nginx/

ln -s /GFPGAN/gfpgan /app/
nginx -g "daemon off;" &

python app.py

