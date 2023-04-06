docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_keras_train:20201010 -f job/tf_keras_train/Dockerfile .
docker push csighub.tencentyun.com/tme-kubeflow/tf_keras_train:20201010

docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_keras_train_ubt:20201010 -f job/tf_keras_train/Dockerfile_ubt .
docker push csighub.tencentyun.com/tme-kubeflow/tf_keras_train_ubt:20201010

docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_keras_train_ubt:20201010-tf260 -f job/tf_keras_train/Dockerfile_ubt_tfnew .
docker push csighub.tencentyun.com/tme-kubeflow/tf_keras_train_ubt:20201010-tf260
