
# Copy values.yaml from desired seldon-core release and modify
# If updating values.yaml
# certManager: true
# Istio.enabled: true
# istio gateway
# kubeflow: true

# Presently need to remove generated status field from resources.yaml file manually

seldon-core-operator/base: clean-kustomize
	mkdir -p seldon-core-operator/base
	cd seldon-core-operator/base && helm template -f ../../values.yaml seldon-core seldon-core-operator --repo https://storage.googleapis.com/seldon-charts --namespace kubeflow --version 1.4.0  > resources.yaml
	#kfctl strangeness means these variables don't always get resolved see https://github.com/kubeflow/manifests/issues/1660
	#sed -i 's#cert-manager.io/inject-ca-from:.*#cert-manager.io/inject-ca-from: $$(CERTIFICATE_NAMESPACE)/$$(CERTIFICATE_NAME)#g' seldon-core-operator/base/resources.yaml
	sed -i 's#cert-manager.io/inject-ca-from:.*#cert-manager.io/inject-ca-from: kubeflow/seldon-serving-cert#g' seldon-core-operator/base/resources.yaml
	sed -i "s#'seldon-webhook-service.kubeflow.svc.cluster.local'#"'$$(SERVICE_NAME).$$(SERVICE_NAMESPACE).svc.cluster.local#' seldon-core-operator/base/resources.yaml
	sed -i "s#commonName: 'seldon-webhook-service.kubeflow.svc'#"'commonName: $$(SERVICE_NAME).$$(CERTIFICATE_NAMESPACE).svc#' seldon-core-operator/base/resources.yaml
	sed -i "s#'seldon-webhook-service.kubeflow.svc'#"'$$(SERVICE_NAME).$$(SERVICE_NAMESPACE).svc#' seldon-core-operator/base/resources.yaml
	sed -i 's#kubeflow/kubeflow-gateway#$$(SERVICE_NAMESPACE)/kubeflow-gateway#' seldon-core-operator/base/resources.yaml
	cp kustomization.tpl seldon-core-operator/base/kustomization.yaml
	cp kustomizeconfig.tpl seldon-core-operator/base/kustomizeconfig.yaml

.PHONY:clean-kustomize
clean-kustomize:
	rm -rf seldon-core-operator/base
