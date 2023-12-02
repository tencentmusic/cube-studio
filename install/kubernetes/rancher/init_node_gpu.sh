
service docker stop
rpm -qa | grep docker | xargs yum remove -y
rpm -qa | grep docker

swapoff -a
rm -rf /data/docker && mkdir -p /data/docker/
yum update -y

yum install docker-ce -y
yum list nvidia-docker2 --showduplicate
yum install -y nvidia-docker2
yum install -y htop docker-compose
yum install -y wireshark

systemctl stop docker


cp -r /var/lib/docker/* /data/docker/
rm -rf /etc/docker/ && mkdir -p /etc/docker/

(
cat << EOF
{
    "insecure-registries":["docker.oa.com:8080"],
    "default-runtime": "nvidia",
    "data-root": "/data/docker",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
)> /etc/docker/daemon.json

systemctl daemon-reload
systemctl start docker

rm -rf /var/lib/docker