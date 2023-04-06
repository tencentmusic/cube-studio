docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_model_evaluation:20201010 -f job/tf_model_evaluation/Dockerfile .
docker push csighub.tencentyun.com/tme-kubeflow/tf_model_evaluation:20201010


docker build --pull -t csighub.tencentyun.com/tme-kubeflow/tf_model_evaluation:20201010-tf260 -f job/tf_model_evaluation/Dockerfile_tfnew .
docker push csighub.tencentyun.com/tme-kubeflow/tf_model_evaluation:20201010-tf260
