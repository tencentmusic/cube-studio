FROM nginx:latest
COPY build/ /usr/share/nginx/html/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN chmod -R 777 /usr/share/nginx/html/