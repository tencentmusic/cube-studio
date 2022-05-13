# !/bin/bash
if [ ! -f "$1" ];then
  echo "[Error] $1 not exist!!!!"
  exit -1
else
  sed 's@SUPPORTED_TENSORFLOW_OPS =@SUPPORTED_TENSORFLOW_OPS = ["\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so"] + @g' $1
  sed -i 's@SUPPORTED_TENSORFLOW_OPS =@SUPPORTED_TENSORFLOW_OPS = ["\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so", "\@recommenders-addons//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so"] + @g' $1
fi
