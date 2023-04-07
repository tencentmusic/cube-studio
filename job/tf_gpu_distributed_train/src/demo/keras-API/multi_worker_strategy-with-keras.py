# Copyright 2018 The Kubeflow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An example of multi-worker training with Keras model using Strategy API."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models


def make_datasets_unbatched():
  BUFFER_SIZE = 10000

  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True,data_dir='/mnt/pengluan/keras/data',download=False)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
  model = models.Sequential()
  model.add(
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))

  model.summary()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model


def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5


def main(args):

  # MultiWorkerMirroredStrategy creates copies of all variables in the model's
  # layers on each device across all workers
  # if your GPUs don't support NCCL, replace "communication" with another
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.NCCL)

  BATCH_SIZE_PER_REPLICA = 64
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

  with strategy.scope():
    ds_train = make_datasets_unbatched().batch(BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
                                        tf.data.experimental.AutoShardPolicy.DATA
    ds_train = ds_train.with_options(options)
    import time
    time.sleep(100000)
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = build_and_compile_cnn_model()

  # Define the checkpoint directory to store the checkpoints
  checkpoint_dir = args.checkpoint_dir

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

  # Function for decaying the learning rate.
  # You can define any decay function you need.
  # Callback for printing the LR at the end of each epoch.
  class PrintLR(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
      print('\nLearning rate for epoch {} is {}'.format(
        epoch + 1, multi_worker_model.optimizer.lr.numpy()))

  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
  ]

  # Keras' `model.fit()` trains the model with specified number of epochs and
  # number of steps per epoch. Note that the numbers here are for demonstration
  # purposes only and may not sufficiently produce a model with good quality.
  multi_worker_model.fit(ds_train,
                         epochs=10,
                         steps_per_epoch=70,
                         callbacks=callbacks)

  # Saving a model
  # Let `is_chief` be a utility function that inspects the cluster spec and
  # current task type and returns True if the worker is the chief and False
  # otherwise.
  def is_chief():
    return (TASK_INDEX == 0)

  if is_chief():
    model_path = args.saved_model_dir

  else:
    # Save to a path that is unique across workers.
    model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)

  multi_worker_model.save(model_path)


if __name__ == '__main__':
  os.environ['NCCL_DEBUG'] = 'INFO'

  tfds.disable_progress_bar()

  # to decide if a worker is chief, get TASK_INDEX in Cluster info
  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
  TASK_INDEX = tf_config['task']['index']

  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_model_dir',
                      type=str,
                      required=True,
                      help='Tensorflow export directory.')

  parser.add_argument('--checkpoint_dir',
                      type=str,
                      required=True,
                      help='Tensorflow checkpoint directory.')

  args = parser.parse_args()
  main(args)
