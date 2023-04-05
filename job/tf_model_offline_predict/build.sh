docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_model_offline_predict:20210426 -f job/tf_model_offline_predict/Dockerfile .
docker push csighub.tencentyun.com/tme-kubeflow/tf_model_offline_predict:20210426


docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_model_offline_predict:20210426-tf260 -f job/tf_model_offline_predict/Dockerfile_tfnew .
docker push csighub.tencentyun.com/tme-kubeflow/tf_model_offline_predict:20210426-tf260
