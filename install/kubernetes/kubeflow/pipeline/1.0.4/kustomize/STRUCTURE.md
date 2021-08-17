# Kubeflow Pipelines Kustomize Manifest Folder Structure

Folder structure:
```
kustomize
├── cluster-scoped-resources
│   ├── README.md (explains this folder structure)
│   └── kustomization.yaml (lists all cluster-scoped folders in base/)
├── base
│   ├── cache-deployer
│   │   ├── cluster-scoped (not included in cache-deployer's kustomization.yaml)
│   │   │   ├── clusteroles
│   │   │   └── clusterrolebindings
|   |   └── ... (namespace scoped)
│   ├── argo
│   │   ├── cluster-scoped
│   │   │   └── workflow-crd.yaml
|   |   └── ... (namespace scoped)
│   ├── application
│   │   ├── cluster-scoped
│   │   │   ├── application-crd.yaml
│   │   │   └── ...
|   |   └── ... (namespace scoped)
│   ├── pipeline
│   │   ├── cluster-scoped
│   │   │   ├── viewer-crd.yaml
│   │   │   ├── scheduledworkflow-crd.yaml
│   │   │   └── ...
|   |   └── ... (namespace scoped)
│   └── ...
└── env
    ├── platform-agnostic
    │   └── kustomization.yaml (based on "base")
    ├── dev
    │   └── kustomization.yaml (based on "env/platform-agnostic")
    └── gcp
        └── kustomization.yaml (based on "base")
```

* User facing manifest entrypoint is `cluster-scoped-resources` package and `env/<env-name>` package.
    * `cluster-scoped-resources` should collect all cluster-scoped resources.
    * `env/<env-name>` should collect env specific namespace scoped resources.
* Universal components live in `base/<component-name>` folders.
    * If a component requires cluster-scoped resources, it should have a folder inside named `cluster-scoped` with related resources, but note that `base/<component-name>/kustomization.yaml` shouldn't include the `cluster-scoped` folder. `cluster-scoped` folders should be collected by top level `cluster-scoped-resources` folder.
* Env specific overlays live in `env/<env-name>` folders.

Constraints we need to comply with (that drove above structure):
* CRDs must be applied separately, because if we apply CRs in the same `kubectl apply` command, the CRD may not have been accepted by k8s api server (e.g. Application CRD).
* [A Kubeflow 1.0 constraint](https://github.com/kubeflow/pipelines/issues/2884#issuecomment-577158715) is that we should separate cluster scoped resources from namespace scoped resources, because sometimes different roles are required to deploy them. Cluster scoped resources usually need a cluster admin role, while namespaced resources can be deployed by individual teams managing a namespace.
