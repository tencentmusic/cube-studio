# docker build -t ccr.ccs.tencentyun.com/cube-studio/fab-upgrade:2020.09.01.0 -f upgrade/Dockerfile .

FROM python:3.6

RUN pip install flask

COPY upgrade /upgrade

WORKDIR /upgrade

CMD ['python','server.py']






