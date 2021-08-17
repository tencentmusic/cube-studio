# This makefile is a quick test to verify all manifests can be hydrated.

test: aws azure dev gcp platform-agnostic platform-agnostic-multi-user

aws: FORCE
	kubectl kustomize env/aws

azure: FORCE
	kubectl kustomize env/azure

dev: FORCE
	kubectl kustomize env/dev

gcp: FORCE
	kubectl kustomize env/gcp

platform-agnostic: FORCE
	kubectl kustomize env/platform-agnostic

platform-agnostic-multi-user: FORCE
	kustomize build --load_restrictor none env/platform-agnostic-multi-user

FORCE: ;
