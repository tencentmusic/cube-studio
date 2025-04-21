version=1.14.1
```bash
wget -c https://github.com/istio/istio/releases/download/$version/istioctl-$version-linux-amd64.tar.gz
tar zxfv istioctl-$version-linux-amd64.tar.gz -C /usr/local/bin/

curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.14.1 TARGET_ARCH=x86_64 sh -

istioctl manifest generate --set Values.global.jwtPolicy=first-party-jwt > install.yaml

kubectl apply -f install.yaml
```

其中主要是base、istiod、gateway三部分

istiod 将先前由 Pilot，Galley，Citadel 和 sidecar 注入器执行的功能统一为一个二进制文件。


# 由1.3.1版本升级到1.4.1+

需要先删除validatingwebhookconfigurations mutatingwebhookconfigurations  deployment和svc ds等

查看所有资源

```bash
namespace=istio-system
kubectl api-resources -o name --verbs=list --namespaced | xargs -n 1 kubectl get --show-kind --ignore-not-found -n $namespace

```
