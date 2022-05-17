# Manifests

This repo contains [kustomize](https://kustomize.io/) packages for deploying Kubeflow applications. 


If you are a contributor authoring or editing the packages please see [Best Practices](./docs/KustomizeBestPractices.md).
Note, please use [kustomize v3.2.1](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv3.2.1) with manifests in this repo, before #538 is fixed which will allow latest kustomize to be used.


# Obsolete information

The information below is obsolete. It pertains to using `kfctl` to generate `kustomization.yaml`. This was how things worked through Kubeflow 1.0.0.

As described in [kubeflow/manifests#1062](https://github.com/kubeflow/manifests/issues/1062) we are working on fixing this in Kubeflow 1.1.0.


## Organization
Subdirectories within the repo hold kustomize targets (base or overlay subdirectory). Overlays contain additional functionality and multiple overlays may be mixed into the base (described below). Both base and overlay targets are processed by kfctl during generate and apply phases and is detailed in [Kfctl Processing](#kfctl-processing).

See [Best Practices](./docs/KustomizeBestPractices.md) for details on how kustomize targets are created.


## Kfctl Processing
Kfctl traverses directories under manifests/kfdef to find and build kustomize targets based on the configuration file `app.yaml`. The contents of app.yaml is the result of running kustomize on the base and specific overlays in the kubeflow/manifests [kfdef](https://github.com/kubeflow/manifests/tree/master/kfdef) directory. The overlays reflect what options are chosen when calling `kfctl init...`.  The kustomize package manager in kfctl will then read app.yaml and apply the packages, components and componentParams to kustomize in the following way:

- **packages**
  - are always top-level directories under the manifests repo
- **components**
  - are also directories but may be a subdirectory in a package.
  - a component may also be a package if there is a base or overlay in the top level directory.
  - otherwise a component is a sub-directory under the package directory.
  - in all cases a component's name in app.yaml must match the directory name.
  - components are output as `<component>.yaml` under the kustomize subdirectory during `kfctl generate...`.
  - in order to output a component, a kustomization.yaml is created above the base or overlay directory and inherits common parameters, namespace and labels of the base or overlay. Additionally it adds the namespace and an application label.
```
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
  - <component>/{base|overlay/<overlay>}
commonLabels:
  app.kubernetes.io/name: <appName>
namespace:
  <namespace>
```
- **component parameters**
  - are applied to a component's params.env file. There must be an entry whose key matches the component parameter. The params.env file is used to generate a ConfigMap. Entries in params.env are resolved as kustomize vars or referenced in a deployment or statefulset env section in which case no var definition is needed.

### Multiple overlays

Kfctl may combine more than one overlay during `kfctl generate ...`. An example is shown below where the profiles target in [manifests](https://github.com/kubeflow/manifests/tree/master/profiles) can include either debug changes in the Deployment or Device information in the Namespace (the devices overlay is not fully integrated with the Profile-controller at this point in time and is intended as an example) or **both**.

```
profiles
├── base
│   └── kustomization.yaml
└── overlays
    ├── debug
    │   └── kustomization.yaml
    └── devices
        └── kustomization.yaml
```

#### What are Multiple Overlays?

Normally kustomize provides the ability to overlay a 'base' set of resources with changes that are merged into the base from resources that are located under an overlays subdirectory. For example
if the kustomize [target](https://github.com/kubernetes-sigs/kustomize/blob/master/docs/glossary.md#target) is named foo there will be a foo/base and possibly one or more overlays such as foo/overlays/bar. A kustomization.yaml file is found in both foo/base and foo/overlays/bar. Running `kustomize build` in foo/base will generate resources as defined in kustomization.yaml. Running `kustomize build` in foo/overlays/bar will generate resources - some of which will overlay the resources in foo/base.

Kustomize doesn't provide for an easy way to combine more than one overlay for example foo/overlays/bar, foo/overlays/baz. However this is an open feature request in kustomize [issues](https://github.com/kubernetes-sigs/kustomize/issues/759). The ability to combine more than one overlay is key to handling components like tf-job-operator which has several types of overlays that can 'mix-in' whether a TFJob is submitted to a namespace or cluster-wide and whether the TFJob uses gang-scheduling.

#### Merging multiple overlays

Since each overlay includes '../../base' as its base set of resources - combining several overlays where each includes '../../base' will cause `kustomize build` to abort, complaining that it recursed on base. The approach is to create a kustomization.yaml at the target level that includes base and the contents of each overlay's kustomization file. This requires some path corrections and some awareness of the behavior of configMapGenerator, secretMapGenerator and how they are copied from each overlay. This kustomization.yaml can be constructed manually, but is integrated within kfctl via the app.yaml file. Using tf-job-operator as an example, if its componentParams has the following
```
  componentParams:
    tf-job-operator:
    - name: overlay
       value: cluster
    - name: overlay
    - value: gangscheduled
```

Then the result will be to combine these overlays eg 'mixin' an overlays in the kustomization.yaml file.

#### Merging multiple overlays to generate app.yaml

In the past when `kfctl init ...` was called it would download the kubeflow repo under `<deployment>/.cache` and read one of the config files under `.cache/kubeflow/<version>/bootstrap/config`. These config files define packages, components and component parameters (among other things). Each config file is a compatible k8 resource of kind *KfDef*. The config files are:

- kfctl_default.yaml
- kfctl_basic_auth.yaml
- kfctl_iap.yaml

Both kfctl_basic_auth.yaml and kfctl_iap.yaml contained the contents of kfctl_default.yaml plus additional changes specific to using kfctl_basic_auth.yaml when --use_basic_auth is passed in or kfctl_iap.yaml when --platform gcp is passed in . This has been refactored to use kustomize where the config/base holds kfctl_default and additional overlays add to the base. The directory now looks like:

```
.
└── config
    ├── base
    │   ├── kfctl_default.yaml
    │   └── kustomization.yaml
    └── overlays
        ├── basic_auth
        │   ├── kfctl_default-patch.yaml
        │   ├── kfctl_default.yaml
        │   └── kustomization.yaml
        ├── gcp
        │   ├── kfctl_default-patch.yaml
        │   ├── kfctl_default.yaml
        │   └── kustomization.yaml
        ├── ksonnet
        │   ├── kfctl_default-patch.yaml
        │   ├── kfctl_default.yaml
        │   └── kustomization.yaml
        └── kustomize
            ├── kfctl_default-patch.yaml
            ├── kfctl_default.yaml
            └── kustomization.yaml
```

Where ksonnet and kustomize hold differing ways of handling the pipeline manifest.

Based on the cli args to `kfctl init...`, the correct overlays will be merged to produce an app.yaml.
The original files have been left as is until UI integration can be completed in a separate PR

### Using kustomize

Generating yaml output for any target can be done using kustomize in the following way:

#### Install kustomize

`go get -u github.com/kubernetes-sigs/kustomize`

### Run kustomize

#### Example

```bash
git clone https://github.com/kubeflow/manifests
cd manifests/<target>/base
kustomize build | tee <output file>
```

Kustomize inputs to kfctl based on app.yaml which is derived from files under kfdef/ such as [kfdef/kfctl_k8s_istio.yaml](https://github.com/kubeflow/manifests/blob/master/kfdef/kfctl_k8s_istio.yaml):

```
apiVersion: kfdef.apps.kubeflow.org/v1
kind: KfDef
metadata:
  namespace: kubeflow
spec:
  applications:
  - kustomizeConfig:
      parameters:
      - name: namespace
        value: istio-system
      repoRef:
        name: manifests
        path: istio/istio-crds
    name: istio-crds
  - kustomizeConfig:
      parameters:
      - name: namespace
        value: istio-system
      repoRef:
        name: manifests
        path: istio/istio-install
    name: istio-install
  - kustomizeConfig:
      parameters:
      - name: clusterRbacConfig
        value: 'OFF'
      repoRef:
        name: manifests
        path: istio/istio
    name: istio
  ......  
    - kustomizeConfig:
      overlays:
      - application
      - istio
      parameters:
      - name: admin
        value: johnDoe@acme.com
      repoRef:
        name: manifests
        path: profiles
    name: profiles
  - kustomizeConfig:
      overlays:
      - application
      repoRef:
        name: manifests
        path: seldon/seldon-core-operator
    name: seldon-core-operator
  repos:
  - name: manifests
    uri: https://github.com/kubeflow/manifests/archive/master.tar.gz
  version: master
```

Outputs from kfctl (no platform specified):
```
kustomize
├── api-service
│   ├── base
│   │   ├── config-map.yaml
│   │   ├── deployment.yaml
│   │   ├── kustomization.yaml
│   │   ├── role-binding.yaml
│   │   ├── role.yaml
│   │   ├── service-account.yaml
│   │   └── service.yaml
│   ├── kustomization.yaml
│   └── overlays
│       └── application
│           ├── application.yaml
│           └── kustomization.yaml
├── argo
│   ├── base
│   │   ├── cluster-role-binding.yaml
│   │   ├── cluster-role.yaml
│   │   ├── config-map.yaml
│   │   ├── crd.yaml
│   │   ├── deployment.yaml
│   │   ├── kustomization.yaml
│   │   ├── params.env
│   │   ├── params.yaml
│   │   ├── service-account.yaml
│   │   └── service.yaml
│   ├── kustomization.yaml
│   └── overlays
│       ├── application
│       │   ├── application.yaml
│       │   └── kustomization.yaml
│       └── istio
│           ├── kustomization.yaml
│           ├── params.yaml
│           └── virtual-service.yaml
......
```

