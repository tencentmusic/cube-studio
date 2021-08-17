# Alpha: Kubeflow on KCC Installation Guide

This instruction explains how to set up Kubeflow on top of Config Connector (KCC) and Anthos Service Mesh (ASM).
Compared with the currently documented GCP deployment, this architecture uses KCC instead of Deployment Manager, and service mesh in the form of ASM instead of open source Istio.

Assume using IAP to protect the kubeflow UI endpoint.

### Benefits of using  KCC

[KCC](https://cloud.google.com/config-connector) is a Google Kubernetes Engine (GKE) addon that allows you to manage your Google Cloud resources through Kubernetes configuration.
With KCC users can manage their Google Cloud infrastructure the same way as manage Kubernetes applications (Infrastructure as code).


### Benefits of using ASM

[ASM](https://cloud.google.com/service-mesh/docs/overview) is a GCP distribution of Istio with more Observability features & Security features.

## Installation Steps


#### Step 0: Setup KCC
If you don't have a running KCC controller yet, follow [KCC instructions](https://cloud.google.com/config-connector/docs/how-to/install-upgrade-uninstall) to create a KCC controller for your organization.
We recommend “Namespaced mode” for KCC controller setup.

From now on assume your KCC controller was hosted in project `kcc-host-project-id`.
Each Project managed by KCC will have a namespace in the KCC cluster named after project id. For example Project “kubeflow-project-id” will linked to a namespace named “kubeflow-project-id” in KCC cluster.
Kfctl | anthoscli | ACP

#### Step 1: Create GCP resources through KCC
* Install kpt

  ```
  gcloud components install kpt alpha
  gcloud components update
  ```

* Set project-id / zone / cluster name

  Checkout latest kubeflow/manifests repo; cd manifests/gcp
  
  Choose a cluster name `export CLUSTER_NAME=choose-name`

  ```
  kpt cfg set v2 gcloud.core.project $(gcloud config get-value project)
  kpt cfg set v2 cluster-name $(CLUSTER_NAME)
  kpt cfg set v2 gcloud.compute.zone $(gcloud config get-value compute/zone)
  ```

* Connect kubectl to KCC cluster

  `gcloud container clusters get-credentials <cluster-name> --zone <> --project <kcc-host-project-id>`

* Apply CNRM resources

  `kustomize build v2/cnrm | kubectl apply -n <kubeflow-project-id> -f -`


#### Step 2: Install ASM
Install ASM on the newly created kubeflow cluster `CLUSTER_NAME`

* Connect kubectl to the new kubeflow cluster `CLUSTER_NAME`

  `gcloud container clusters get-credentials $(CLUSTER_NAME) --zone <> --project <kubeflow-project-id>`

* [Set credentials and permissions](https://cloud.google.com/service-mesh/docs/gke-install-existing-cluster#set_credentials_and_permissions)

* [Download istioctl released by GCP](https://cloud.google.com/service-mesh/docs/gke-install-existing-cluster#download_the_installation_file)

* Run Istioctl (download in previous step)

  `istioctl manifest apply -f v2/asm/istio-operator.yaml`

	
#### Step 3: Deploy Kubeflow components

* [Setup Environment Variables for IAP](https://www.kubeflow.org/docs/gke/deploy/oauth-setup/)
	
	```
	export CLIENT_ID=
	export CLIENT_SECRET=
  ```

* Install Kubeflow on the newly created cluster

  ```
  mkdir $(CLUSTER_NAME) && cd $(CLUSTER_NAME)
  kfctl apply -V -f https://raw.githubusercontent.com/kubeflow/manifests/master/kfdef/kfctl_gcp_asm_exp.yaml
  ```