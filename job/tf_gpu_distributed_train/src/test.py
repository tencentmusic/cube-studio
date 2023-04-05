import tensorflow as tf
print(tf.__version__)
print(tf.__path__)

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

from tensorflow.python.client import device_lib

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
# 打印
#     print(local_device_protos)

# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']


