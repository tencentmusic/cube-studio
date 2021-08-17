
FROM tensorflow/serving:1.14.0
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

RUN apt update && apt install -y --force-yes --no-install-recommends python3.6-dev python3-pip vim ca-certificates-java jq  wget dnsutils iputils-ping net-tools curl locales ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy && \
    locale-gen zh_CN && locale-gen zh_CN.utf8 && \ 
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip


