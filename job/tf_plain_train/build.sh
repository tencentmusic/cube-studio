docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_plain_train:20201010 -f job/tf_plain_train/Dockerfile .
docker push csighub.tencentyun.com/tme-kubeflow/tf_plain_train:20201010


docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_plain_train_ubt:20201010 -f job/tf_plain_train/Dockerfile_ubt .
docker push csighub.tencentyun.com/tme-kubeflow/tf_plain_train_ubt:20201010

docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_plain_train:20201010-tf260 -f job/tf_plain_train/Dockerfile_tfnew .
docker push csighub.tencentyun.com/tme-kubeflow/tf_plain_train:20201010-tf260


docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_plain_train_ubt:20201010-tf260 -f job/tf_plain_train/Dockerfile_ubt_tfnew .
docker push csighub.tencentyun.com/tme-kubeflow/tf_plain_train_ubt:20201010-tf260
