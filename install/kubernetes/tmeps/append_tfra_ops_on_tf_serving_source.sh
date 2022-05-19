# !/bin/bash
model_servers_build_file=/tensorflow-serving/tensorflow_serving/model_servers/BUILD
if [ ! -f "$model_servers_build_file" ];then
  echo "[Error] $model_servers_build_file not exist !!"
  exit -1
fi

sed 's@SUPPORTED_TENSORFLOW_OPS =@SUPPORTED_TENSORFLOW_OPS = ["\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so"] + @g' $model_servers_build_file
sed -i 's@SUPPORTED_TENSORFLOW_OPS =@SUPPORTED_TENSORFLOW_OPS = ["\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so"] + @g' $model_servers_build_file

workspace_file=/tensorflow-serving/WORKSPACE
if [ ! -f "$workspace_file" ];then
  echo "[Error] $workspace_file not exist !!"
  exit -1
fi

echo '
local_repository(
    name = "recommenders-addons",
    path = "../recommenders-addons",
)' >> $workspace_file

cat /recommenders-addons/WORKSPACE | grep -v 'workspace(name = "tf_recommenders_addons")' | sed 's@load("//@load("\@recommenders-addons//@g' | sed 's@build_file = "//@build_file = "\@recommenders-addons//@g' >> $workspace_file

bazelrc_file=/tensorflow-serving/.bazelrc

if [ ! -f "$bazelrc_file" ];then
  echo "[Error] $bazelrc_file not exist !!"
  exit -1
fi

echo '
build --action_env TF_HEADER_DIR="/usr/local/lib/python3.6/dist-packages/tensorflow/include"
build --action_env TF_SHARED_LIBRARY_DIR="/usr/local/lib/python3.6/dist-packages/tensorflow"
build --action_env TF_VERSION_INTEGER="2051"
build --action_env TF_SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"
build --action_env TF_CXX11_ABI_FLAG="0"
build --action_env FOR_TF_SERVING="1"
build --spawn_strategy=standalone
build --strategy=Genrule=standalone
build -c opt 
#build -c dbg
#build --copt=-mavx
build --action_env TF_NEED_CUDA="0"

#build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda"
#build --action_env CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"
#build --action_env TF_CUDA_VERSION="11.2"
#build --action_env TF_CUDNN_VERSION="8"

#test --config=cuda
#build --config=cuda
#build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true
#build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain

build --linkopt=-Wl,--hash-style=sysv' >> $bazelrc_file
