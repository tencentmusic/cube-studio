version=1.14.1
wget -c https://github.com/istio/istio/releases/download/$version/istioctl-$version-linux-amd64.tar.gz
tar zxfv istioctl-$version-linux-amd64.tar.gz -C /usr/local/bin/

curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.14.1 TARGET_ARCH=x86_64 sh -

istioctl manifest generate --set Values.global.jwtPolicy=first-party-jwt > install.yaml

kubectl apply -f install.yaml

其中主要是base、istiod、gateway三部分

istiod 将先前由 Pilot，Galley，Citadel 和 sidecar 注入器执行的功能统一为一个二进制文件。

