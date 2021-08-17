# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import base64

kfp_version = os.environ["KFP_VERSION"]
disable_istio_sidecar = os.environ.get("DISABLE_ISTIO_SIDECAR") == "true"
mlpipeline_minio_access_key = base64.b64encode(
    bytes(os.environ.get("MINIO_ACCESS_KEY"), 'utf-8')).decode('utf-8')
mlpipeline_minio_secret_key = base64.b64encode(
    bytes(os.environ.get("MINIO_SECRET_KEY"), 'utf-8')).decode('utf-8')


class Controller(BaseHTTPRequestHandler):
    def sync(self, parent, children):
        # HACK: Currently using serving.kubeflow.org/inferenceservice to identify
        # kubeflow user namespaces.
        # TODO: let Kubeflow profile controller add a pipeline specific label to
        # user namespaces and use that label instead.
        pipeline_enabled = parent.get("metadata", {}).get(
            "labels", {}).get("serving.kubeflow.org/inferenceservice")

        if not pipeline_enabled:
            return {"status": {}, "children": []}

        # Compute status based on observed state.
        desired_status = {
            "kubeflow-pipelines-ready": \
                len(children["Secret.v1"]) == 1 and \
                len(children["ConfigMap.v1"]) == 1 and \
                len(children["Deployment.apps/v1"]) == 2 and \
                len(children["Service.v1"]) == 2 and \
                len(children["DestinationRule.networking.istio.io/v1alpha3"]) == 1 and \
                len(children["ServiceRole.rbac.istio.io/v1alpha1"]) == 1 and \
                len(children["ServiceRoleBinding.rbac.istio.io/v1alpha1"]) == 1 and \
                "True" or "False"
        }

        # Generate the desired child object(s).
        # parent is a namespace
        namespace = parent.get("metadata", {}).get("name")
        desired_resources = [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "metadata-grpc-configmap",
                    "namespace": namespace,
                },
                "data": {
                    "METADATA_GRPC_SERVICE_HOST":
                    "metadata-grpc-service.kubeflow",
                    "METADATA_GRPC_SERVICE_PORT": "8080",
                },
            },
            # Visualization server related manifests below
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "labels": {
                        "app": "ml-pipeline-visualizationserver"
                    },
                    "name": "ml-pipeline-visualizationserver",
                    "namespace": namespace,
                },
                "spec": {
                    "selector": {
                        "matchLabels": {
                            "app": "ml-pipeline-visualizationserver"
                        },
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "ml-pipeline-visualizationserver"
                            },
                            "annotations": disable_istio_sidecar and {
                                "sidecar.istio.io/inject": "false"
                            } or {},
                        },
                        "spec": {
                            "containers": [{
                                "image":
                                "gcr.io/ml-pipeline/visualization-server:" +
                                kfp_version,
                                "imagePullPolicy":
                                "IfNotPresent",
                                "name":
                                "ml-pipeline-visualizationserver",
                                "ports": [{
                                    "containerPort": 8888
                                }],
                            }],
                            "serviceAccountName":
                            "default-editor",
                        },
                    },
                },
            },
            {
                "apiVersion": "networking.istio.io/v1alpha3",
                "kind": "DestinationRule",
                "metadata": {
                    "name": "ml-pipeline-visualizationserver",
                    "namespace": namespace,
                },
                "spec": {
                    "host": "ml-pipeline-visualizationserver",
                    "trafficPolicy": {
                        "tls": {
                            "mode": "ISTIO_MUTUAL"
                        }
                    }
                }
            },
            {
                "apiVersion": "rbac.istio.io/v1alpha1",
                "kind": "ServiceRole",
                "metadata": {
                    "name": "ml-pipeline-visualizationserver",
                    "namespace": namespace,
                },
                "spec": {
                    "rules": [{
                        "services": ["ml-pipeline-visualizationserver.*"]
                    }]
                }
            },
            {
                "apiVersion": "rbac.istio.io/v1alpha1",
                "kind": "ServiceRoleBinding",
                "metadata": {
                    "name": "ml-pipeline-visualizationserver",
                    "namespace": namespace,
                },
                "spec": {
                    "subjects": [{
                        "properties": {
                            "source.principal":
                            "cluster.local/ns/kubeflow/sa/ml-pipeline"
                        }
                    }],
                    "roleRef": {
                        "kind": "ServiceRole",
                        "name": "ml-pipeline-visualizationserver"
                    }
                }
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "ml-pipeline-visualizationserver",
                    "namespace": namespace,
                },
                "spec": {
                    "ports": [{
                        "name": "http",
                        "port": 8888,
                        "protocol": "TCP",
                        "targetPort": 8888,
                    }],
                    "selector": {
                        "app": "ml-pipeline-visualizationserver",
                    },
                },
            },
            # Artifact fetcher related resources below.
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "labels": {
                        "app": "ml-pipeline-ui-artifact"
                    },
                    "name": "ml-pipeline-ui-artifact",
                    "namespace": namespace,
                },
                "spec": {
                    "selector": {
                        "matchLabels": {
                            "app": "ml-pipeline-ui-artifact"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "ml-pipeline-ui-artifact"
                            },
                            "annotations": disable_istio_sidecar and {
                                "sidecar.istio.io/inject": "false"
                            } or {},
                        },
                        "spec": {
                            "containers": [{
                                "name":
                                "ml-pipeline-ui-artifact",
                                "image":
                                "gcr.io/ml-pipeline/frontend:" + kfp_version,
                                "imagePullPolicy":
                                "IfNotPresent",
                                "ports": [{
                                    "containerPort": 3000
                                }]
                            }],
                            "serviceAccountName":
                            "default-editor"
                        }
                    }
                }
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "ml-pipeline-ui-artifact",
                    "namespace": namespace,
                    "labels": {
                        "app": "ml-pipeline-ui-artifact"
                    }
                },
                "spec": {
                    "ports": [{
                        "name":
                        "http",  # name is required to let istio understand request protocol
                        "port": 80,
                        "protocol": "TCP",
                        "targetPort": 3000
                    }],
                    "selector": {
                        "app": "ml-pipeline-ui-artifact"
                    }
                }
            },
        ]
        print('Received request:', parent)
        print('Desired resources except secrets:', desired_resources)
        # Moved after the print argument because this is sensitive data.
        desired_resources.append({
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "mlpipeline-minio-artifact",
                "namespace": namespace,
            },
            "data": {
                "accesskey": mlpipeline_minio_access_key,
                "secretkey": mlpipeline_minio_secret_key,
            },
        })

        return {"status": desired_status, "children": desired_resources}

    def do_POST(self):
        # Serve the sync() function as a JSON webhook.
        observed = json.loads(
            self.rfile.read(int(self.headers.get("content-length"))))
        desired = self.sync(observed["parent"], observed["children"])

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(desired), 'utf-8'))


HTTPServer(("", 80), Controller).serve_forever()
