# KFP customizations for Azure

This template provides a starting point to configure KFP to use an Azure hosted MySQL database, as well as an Azure Blob backed MinIO service.

## MySQL

1. [Create an Azure Database for MySQL](https://docs.microsoft.com/azure/mysql/quickstart-create-mysql-server-database-using-azure-portal). Ensure that it will allow connections from the Kubernetes cluster.

2. Substitute the server name into [params.env](./params.env), and the username and password into [mysql-secret.env](./mysql-secret.env)

## MinIO Gateway for Azure Blobstore

1. [Create an Azure Storage account](https://docs.microsoft.com/azure/storage/common/storage-account-create). Ensure that it will allow connections from the Kubernetes cluster.

2. Substitute the storage name and access key into [minio-artifact-secret.env](./minio-azure-gateway/minio-artifact-secret.env).
