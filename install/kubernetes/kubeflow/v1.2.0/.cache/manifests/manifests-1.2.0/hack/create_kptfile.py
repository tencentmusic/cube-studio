"""This is a script created for the updating of the GCP Kpt packages.

The purpose of this is to upgrade to using setters and substitutions
"""

import fire
import logging
import os
import re
import subprocess

def create_setter(name, value, cwd, field=None):
  command = ["kpt", "cfg", "create-setter", ".", name, value]

  if field:
    command.append("--field")
    command.append(field)

  logging.info("Run:\n" + " ".join(command))
  subprocess.check_call(command,  cwd=cwd)

def create_subst(name, value, pattern, cwd):
  command = ["kpt", "cfg", "create-subst", ".",  name,
             "--field-value", value,
             "--pattern", pattern]
  logging.info("Run:\n" + " ".join(command))
  subprocess.check_call(command,  cwd=cwd)

class KptCreator:
  @staticmethod
  def strip_comments(path):
    """Strip the existing comments from YAML files"""

    for root, _, files in os.walk(path):
      for f in files:
        ext = os.path.splitext(f)[-1]
        logging.info(f"{ext}")
        if ext != ".yaml":
          continue

        p = os.path.join(root, f)
        logging.info(f"Proccessing {p}")

        with open(p) as hf:
          lines = hf.readlines()

        new_lines = []

        for l in lines:
          if re.match("[^#]+#.*x-kustomize.*", l):
            pieces = l.split("#", 1)
            new_lines.append(pieces[0].rstrip() + "\n")
          else:
            new_lines.append(l)

        with open(p, "w") as hf:
          hf.writelines(new_lines)

  @staticmethod
  def create_subst_asm(path):
    #
    create_setter("gcloud.project.projectNumber", "147474701642", path)

    create_subst("asm-gcp-metadata", "project-id|147474701642|name|us-central1-c",
                 "${gcloud.core.project}|${gcloud.project.projectNumber}|${name}|${location}", path)

    create_subst("asm-cluster-url", "https://container.googleapis.com/v1/projects/project-id/locations/us-central1/clusters/name",
                 "https://container.googleapis.com/v1/projects/${gcloud.core.project}/locations/${location}/clusters/${name}",
                 path)

    create_subst("asm-mesh-id", "project-id_us-central1_name",
                 "${gcloud.core.project}_${location}_${name}", path)

    create_subst("mesh-id", "project-id_us-east1-d_name",
                 "${gcloud.core.project}_${location}_${name}", path)

    create_subst("asm-cluster-name", "project-id/us-central1/name",
                 "${gcloud.core.project}/${location}/${name}", path)


  @staticmethod
  def create_subst_private(path):
    registries = ["quay.io/jetstack", "gcr.io/kubeflow-images-public",
                  "metacontroller"]
    for registry in registries:
      n = registry.replace("/", ".")
      name = f"image-mirror-{n}"
      value = f"gcr.io/project-id/mirror/{registry}"
      pattern = f"gcr.io/${{gcloud.core.project}}/mirror/{registry}"

      create_subst(name, value, pattern, path)

  @staticmethod
  def create_subst_mgmt(path):
    # Management cluster is using us-central1
    create_setter("location", "us-central1-f", path)
    create_setter("name", "name", path)
    create_setter("gcloud.core.project", "project", path)

    create_subst("cluster-name",
                 "project-id/mgmt-location/mgmt-name",
                 "${gcloud.core.project}/${location}/${name}", path)

    create_subst("pool",
                 "name-pool",
                 "${name}-pool", path)

    create_subst(
      "cnrm-sa",
      "serviceAccount:mgmt-project-id.svc.id.goog[cnrm-system/cnrm-controller-manager]",
      "serviceAccount:${gcloud.core.project}.svc.id.goog[cnrm-system/cnrm-controller-manager]",
      path)

  @staticmethod
  def create_subst_stacks(path):
    create_setter("name", "name", path)
    create_setter("gcloud.core.project", "project-id", path)

     # Admin service account ref
    create_subst("admin-sa-ref",
                 "name-admin@project-id.iam.gserviceaccount.com",
                 "${name}-admin@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    # User service account ref
    create_subst("user-sa-ref",
                 "name-user@project-id.iam.gserviceaccount.com",
                 "${name}-user@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)


  @staticmethod
  def create_subst(path):
    # Service account substitutions
    create_setter("gcloud.core.project", "project-id", path)
    create_setter("name", "name", path)

    create_setter("gcloud.compute.zone", "us-east1-d", path)
    create_setter("gcloud.compute.region", "us-central1", path)

    # Workload identity bindings for the kf-admin account
    for ns in ["kubeflow", "istio-system"]:
      name = f"iampolicy-member-kfadmin-{ns}"
      value = f"serviceAccount:project-id.svc.id.goog[{ns}/kf-admin]"
      pattern = f"serviceAccount:${{gcloud.core.project}}.svc.id.goog[{ns}/kf-admin]"
      create_subst(name, value, pattern, path)


    # For user account create names for IAM policy member rules
    services = ["cloudbuild", "viewer", "source",
                "storage", "bigquery", "dataflow",
                "ml", "dataproc", "cloudsql", "logging",
                "metricwriter", "monitoringviewer"]


    # Import create zone and region before location so that location overrides
    # it
    create_setter("location", "us-east1-d", path)

    # Private GKE
    create_setter("log-firewalls", "false", path, field="spec.enableLogging")


    create_subst("name-storage-metadata-store", "name-storage-metadata-store",
                 "${name}-storage-metadata-store", path)
    create_subst("name-storage-artifact-store", "name-storage-artifact-store",
                 "${name}-storage-artifact-store", path)
    create_subst("name-ip", "name-ip", "${name}-ip", path)

    # DNS
    zones = ["gcr", "gcr-cname", "gcr-a", "goog-apis", "goog-cname",
             "goog-a"]
    for z in zones:
      name = f"name-{z}"
      value = f"name-{z}"
      pattern = f"${{name}}-{z}"
      create_subst(name, value, pattern, path)

    # Routes:
    routes = ["google-apis", "internet"]
    for r in routes:
      name = f"name-{r}"
      value = f"name-{r}"
      pattern= f"${{name}}-{r}"
      create_subst(name, value, pattern, path)

    # Names of firewall rules
    rules = ["deny-egress", "health-ingress", "health-egress", "apis-egress",
             "master-egress", "int-egress", "istio", "cm", "dockerhub",
             "iap-jwks"]

    for r in rules:
      name = f"name-{r}"
      value = f"name-{r}"
      pattern= f"${{name}}-{r}"
      create_subst(name, value, pattern, path)


    # Names for IAM Policies granting pipelines KSA's workload identity
    # on user service account
    ksa_names = ["ml-pipeline-ui",
                 "ml-pipeline-visualization", # TODO(jlewi): Not sure we actually need this.
                 "ml-pipeline-visualizationserver",
                 "pipeline-runner"]

    for ksa in ksa_names:
      name = f"name-user-workload-identity-user-{ksa}"
      value = f"name-user-workload-identity-user-{ksa}"
      pattern = "${name}-user-workload-identity-user-" + f"{ksa}"
      create_subst(name, value, pattern, path)

    # Members for IAM policy members for these service account
    for ksa in ksa_names:
      name = f"name-user-workload-identity-user-{ksa}-member"
      value = f"serviceAccount:project-id.svc.id.goog[kubeflow/{ksa}]"
      pattern = f"serviceAccount:${{gcloud.core.project}}.svc.id.goog[kubeflow/{ksa}]"
      create_subst(name, value, pattern, path)


    # For user account create names for IAM policy member rules
    services = ["cloudbuild", "viewer", "source",
                "storage", "bigquery", "dataflow",
                "ml", "dataproc", "cloudsql", "logging",
                "metricwriter", "monitoringviewer"]

    for s in services:
      name = f"name-user-{s}"
      value = f"name-user-{s}"
      pattern = "${name}-user-" + f"{s}"
      create_subst(name, value, pattern, path)

    # For vm account create substitutions of names of IAM policy members
    create_subst("name-vm-policy-logging", "name-vm-logging",
                 "${name}-vm-logging", path)

    policies = ["monitoring", "meshtelemetry", "cloudtrace",
                "monitoring-viewer", "storage"]

    for a in policies:
      name = f"name-vm-policy-{a}"
      value = f"name-vm-policy-{a}"
      pattern = "${name}-vm-policy-" + f"{a}"
      create_subst(name, value, pattern, path)


    # Cluster substitutions
    create_subst("cluster-name", "project-id/us-east1-d/name",
                 "${gcloud.core.project}/${location}/${name}", path)

    create_subst("identity-ns", "project-id.svc.id.goog",
                 "${gcloud.core.project}.svc.id.goog", path)

    # Names for service accounts
    create_subst("admin-sa-name",
                 "name-admin",
                 "${name}-admin",
                 path)

    create_subst("user-sa-name",
                 "name-user",
                 "${name}-user",
                 path)

    # Workload identity
    create_subst("name-admin-wi", "name-admin-workload-identity-user",
                 "${name}-admin-workload-identity-user", path)

    create_subst("admin-profiles-sa-wi",
                 "serviceAccount:project-id.svc.id.goog[kubeflow/profiles-controller-service-account]",
                 "serviceAccount:${gcloud.core.project}.svc.id.goog[kubeflow/profiles-controller-service-account]",
                 path)

    # Names for WI identity bindings
    for suffix in ["ml-pipeline-ui", "ml-pipeline-visualizationserver", "pipeline-runner"]:
      name = "user-wi-" + suffix
      value = "name-user-workload-identity-user-" + suffix
      pattern = "${name}" + "-user-workload-identity-user-" + suffix

      create_subst(name, value, pattern, path)

    create_subst("projects",
                 "projects/project-id",
                 "projects/${gcloud.core.project}",
                 path)

    create_subst("admin-service-account",
                 "serviceAccount:name-admin@project-id.iam.gserviceaccount.com",
                 "serviceAccount:${name}-admin@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    create_subst("user-service-account",
                 "serviceAccount:name-user@project-id.iam.gserviceaccount.com",
                 "serviceAccount:${name}-user@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    create_subst("vm-service-account",
                 "serviceAccount:name-vm@project-id.iam.gserviceaccount.com",
                 "serviceAccount:${name}-vm@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    # VM Service account ref
    create_subst("vm-sa-ref",
                 "name-vm@project-id.iam.gserviceaccount.com",
                 "${name}-vm@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    # Admin service account ref
    create_subst("admin-sa-ref",
                 "name-admin@project-id.iam.gserviceaccount.com",
                 "${name}-admin@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)

    # User service account ref
    create_subst("user-sa-ref",
                 "name-user@project-id.iam.gserviceaccount.com",
                 "${name}-user@${gcloud.core.project}.iam.gserviceaccount.com",
                 path)


    create_subst("node-pool-cpu",
                 "name-cpu-pool-v1",
                 "${name}-cpu-pool-v1",
                 path)


    create_subst("name-admin-manages-user",
                 "name-admin-manages-user","${name}-admin-manages-user", path)

    # Create policy substitutions for admin account
    policies = ["admin-source", "admin-servicemanagement", "admin-network",
                "admin-cloudbuild", "admin-viewer", "admin-storage", "admin-bigquery",
                "admin-dataflow", "admin-ml", "admin-dataproc", "admin-cloudsql",
                "admin-logging", "admin-metricwriter",
                "admin-monitoringviewer",]

    for a in policies:
      create_subst(a + "-iam", f"name-{a}", r"""${name}-""" + a, path)

    create_subst("name-vm", "name-vm", "${name}-vm", path)

    KptCreator.create_subst_asm(path)
    KptCreator.create_subst_private(path)
    KptCreator.restore()

  @staticmethod
  def restore():
    subdirs = ["gcp/cloud-endpoints", "gcp/deployment_manager_configs",
               "gcp/gpu-driver", "gcp/iap-ingress",
               "gcp/prometheus", "gcp/privateutil"]
    for subdir in subdirs:
      subprocess.check_call(["git", "checkout", "upstream/master", subdir])
if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format=('%(levelname)s|%(asctime)s'
              '|%(pathname)s|%(lineno)d| %(message)s'),
      datefmt='%Y-%m-%dT%H:%M:%S',
  )
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(KptCreator)
