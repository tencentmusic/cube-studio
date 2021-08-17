REPO_ROOT=../../../..

update:
	rm -rf upstream
	mkdir upstream
	# This currently yields an error after fetching upstream manifests:
	# > wrong Node Kind for  expected: MappingNode was SequenceNode
	# The error can be ignored for now.
	# TODO(Bobgy): figure out a workaround.
	-kpt pkg get "https://github.com/argoproj/argo-workflows.git/manifests@$$(cat $(REPO_ROOT)/third_party/argo/VERSION)" upstream/
	# Remove the pre-hydrated manifests which we do not use.
	rm upstream/manifests/*.yaml
	# Remove README.md which might be confusing here.
	rm upstream/manifests/README.md
