

# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/istio:proxyv2-1.14.1 .
FROM docker.io/istio/proxyv2:1.14.1
USER root
RUN apt update -y
RUN apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl mysql-client locales zip traceroute
